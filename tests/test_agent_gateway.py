import json
from pathlib import Path
import tempfile
import threading
import unittest
import wave
import struct
from unittest.mock import patch
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from app import agent_service as mod


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.gateway = mod.Gateway(self.root / "state", idle_seconds=1)
        self.media = self.root / "source.mp4"
        self.media.write_bytes(b"test")
        self.reference = self.root / "ref.wav"
        self.reference.write_bytes(b"reference")
        self.gateway.add_voice({"id": "a", "reference": str(self.reference)})

    def tearDown(self):
        self.temp.cleanup()

    def submit(self, **extra):
        data = {"input": str(self.media), "voice": "a", **extra}
        return self.gateway.jobs[self.gateway.submit(data)["id"]]

    def test_idempotency_is_atomic_and_conflicting_key_rejected(self):
        request = {"input": str(self.media), "voice": "a"}
        a = self.gateway.submit(request, "stable")
        b = self.gateway.submit(request, "stable")
        self.assertEqual(a["id"], b["id"])
        with self.assertRaises(mod.Problem):
            self.gateway.submit(dict(request, background="keep"), "stable")

    def test_requests_without_idempotency_key_are_distinct(self):
        self.assertNotEqual(self.submit()["id"], self.submit()["id"])

    def test_workspace_keeps_configuration_status_events_and_outputs_together(self):
        workspace = self.root / "project" / "agent_work"
        job = self.submit(workspace=str(workspace))
        self.assertEqual(job["request"]["output"], str(workspace / "04_outputs"))
        self.assertTrue((workspace / "01_config" / (job["id"] + ".json")).is_file())
        self.gateway.update(job, status="needs_review")
        snapshot = workspace / "05_logs" / job["id"] / "status.json"
        self.assertEqual(json.loads(snapshot.read_text())["status"], "needs_review")
        self.assertEqual(len((snapshot.parent / "events.jsonl").read_text().splitlines()), 2)
        self.assertTrue((workspace / "99_tmp" / job["id"]).is_dir())
        restored = mod.Gateway(self.gateway.state)
        self.assertEqual(restored.get(job["id"])["request"]["workspace"], str(workspace))

    def test_invalid_numbers_and_unknown_parameters(self):
        for patch_data in ({"speed": float("nan")}, {"speed": True}, {"speaker_num": 1.5},
                           {"max_retries": 4}, {"seed": -1}, {"speed": 5}, {"fit_to_slot": "false"},
                           {"temperature": .8}):
            with self.subTest(patch_data=patch_data), self.assertRaises(mod.Problem):
                self.submit(**patch_data)

    def test_voice_is_copied_immutable_and_path_traversal_rejected(self):
        self.reference.unlink()
        self.assertTrue(Path(self.gateway.voice("a")["reference"]).is_file())
        with self.assertRaises(mod.Problem):
            self.gateway.voice("../x")

    def test_restart_preserves_queue_and_marks_running_interrupted(self):
        queued = self.submit()
        running = self.submit()
        self.gateway.update(running, status="running")
        restored = mod.Gateway(self.gateway.state)
        self.assertEqual(restored.get(queued["id"])["status"], "queued")
        self.assertEqual(restored.get(running["id"])["status"], "interrupted")

    def test_cancel_is_only_for_queued_jobs(self):
        job = self.submit()
        self.assertEqual(self.gateway.action(job["id"], "cancel", {})["status"], "cancelled")
        with self.assertRaises(mod.Problem):
            self.gateway.action(job["id"], "cancel", {})

    def fake_backend(self, multi=False, unknown=False, mismatch=False):
        self.calls = []
        segments = [{"segment_id": "s1", "speaker_id": "speaker_0", "text": "你好"}]
        if multi:
            segments.append({"segment_id": "s2", "speaker_id": "speaker_1", "text": "再见"})
        if unknown:
            segments.append({"segment_id": "s3", "speaker_id": "unknown", "text": "好"})
        self.timeline = {"duration_sec": 1, "speakers": [{"speaker_id": "speaker_0"}, {"speaker_id": "speaker_1"}], "segments": segments}
        audio = self.root / "generated.wav"
        with wave.open(str(audio), "wb") as out:
            out.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            out.writeframes(struct.pack("<h", 3000) * 24000)
        self.generated = False

        def fake(method, route, payload=None, **kwargs):
            self.calls.append((method, route, payload, kwargs))
            if route == "/api/system/lease":
                return {"token": "lease"}
            if route.endswith("/analyze"):
                return {"job_id": "child"}
            if route.endswith("/timeline"):
                if method == "POST":
                    for edit in payload["edits"]:
                        next(s for s in segments if s["segment_id"] == edit["segment_id"]).update(edit)
                return json.loads(json.dumps(self.timeline))
            if route.endswith("/generate"):
                self.generated = True
                return {}
            if route.endswith("/jobs/child"):
                return {"status": "done", "quality": {"segment_count": 0 if mismatch else len(segments)},
                        "generation_urls": {"video": "/video.mp4"},
                        "generation": {"video": str(self.media), "voice_track": str(audio), "mix_track": str(audio)}}
            return {}
        return fake

    def execute(self, job, **kwargs):
        with patch.object(self.gateway, "start", return_value={}), patch.object(mod, "backend", self.fake_backend(**kwargs)), patch.object(mod, "media_metrics", return_value={"valid": True}):
            self.gateway.execute(job)

    def test_single_voice_unknown_is_explicitly_reassigned_and_mix_removes_original(self):
        job = self.submit()
        self.execute(job, unknown=True)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["quality"]["unknown_reassigned"], 1)
        controls = next(c[2] for c in self.calls if c[1].endswith("/generate"))
        self.assertEqual([controls[k] for k in ("target_original_volume", "other_speaker_volume", "background_volume")], [0, 0, 0])

    def test_multi_voice_missing_mapping_pauses_before_generation(self):
        job = self.submit()
        job["request"].pop("voice")
        job["request"]["voices"] = {"speaker_0": "a"}
        self.execute(job, multi=True)
        self.assertEqual(job["status"], "needs_review")
        self.assertFalse(self.generated)

    def test_multi_voice_unknown_pauses_and_resume_accepts_patch(self):
        job = self.submit()
        job["request"].pop("voice")
        job["request"]["voices"] = {"speaker_0": "a"}
        self.execute(job, unknown=True)
        self.assertEqual(job["missing_speakers"], ["unknown"])
        self.gateway.action(job["id"], "resume", {"unknown_speaker_id": "speaker_0"})
        self.assertEqual(job["status"], "queued")

    def test_agent_can_review_transcript_before_generation(self):
        job = self.submit(review_after_analysis=True)
        self.execute(job)
        self.assertEqual(job["status"], "needs_review")
        self.assertFalse(self.generated)
        self.gateway.action(job["id"], "resume", {})
        self.assertTrue(job["analysis_reviewed"])

    def test_segment_mismatch_is_not_success(self):
        job = self.submit()
        self.execute(job, mismatch=True)
        self.assertEqual(job["status"], "needs_review")

    def test_silent_generated_track_is_flagged(self):
        job = self.submit()
        fake = self.fake_backend()
        with wave.open(str(self.root / "generated.wav"), "wb") as out:
            out.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            out.writeframes(b"\0\0" * 24000)
        with patch.object(self.gateway, "start", return_value={}), patch.object(mod, "backend", fake), patch.object(mod, "media_metrics", return_value={"valid": True}):
            self.gateway.execute(job)
        self.assertEqual(job["status"], "needs_review")
        self.assertTrue(any("静音" in warning for warning in job["quality"]["warnings"]))

    def test_owned_services_remain_managed_after_gateway_restart(self):
        mod.atomic_json(self.gateway.state / "runtime.json", {"managed": True})
        self.assertTrue(mod.Gateway(self.gateway.state).managed)

    def test_keep_background_and_independent_speaker_bindings(self):
        job = self.submit(background="keep", voices={"speaker_1": "a"})
        self.execute(job, multi=True)
        controls = next(c[2] for c in self.calls if c[1].endswith("/generate"))
        self.assertEqual(controls["background_volume"], 1)
        self.assertEqual(set(job["resolved_voices"]), {"speaker_0", "speaker_1"})

    def test_stop_refuses_queue_and_busy_ui(self):
        self.submit()
        with self.assertRaises(mod.Problem):
            self.gateway.stop()
        self.gateway.jobs.clear()
        with patch.object(mod, "run", return_value="true"), patch.object(mod, "backend", return_value={"idle": False}):
            with self.assertRaises(mod.Problem):
                self.gateway.stop()

    def test_stop_targets_only_two_containers_and_does_not_delete_images(self):
        calls = []
        def fake_run(*args, **kwargs):
            calls.append(args)
            return "true"
        with patch.object(mod, "run", fake_run), patch.object(mod, "backend", return_value={"idle": True}):
            self.assertEqual(self.gateway.stop()["status"], "stopped")
        stop = next(c for c in calls if c[:2] == ("docker", "stop"))
        self.assertEqual(stop[-2:], mod.CONTAINERS)
        self.assertFalse(any("rm" in c for c in calls))

    def test_http_file_download_and_origin_rejection(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
        server.gateway, server.token = self.gateway, ""
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:" + str(server.server_port)
        try:
            request = urllib.request.Request(base + "/api/agent/jobs", data=b"{}", headers={"Origin": "http://evil.example"})
            with self.assertRaises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(request)
            self.assertEqual(exc.exception.code, 403)
            exc.exception.close()
            job = self.submit()
            self.gateway.update(job, result={"generation": {"video": str(self.media)}})
            with urllib.request.urlopen(base + "/api/agent/jobs/" + job["id"] + "/files/video") as response:
                self.assertEqual(response.read(), b"test")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
