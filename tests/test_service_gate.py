"""Run inside the workbench image, with VOICE_STATE_DIR set to a temporary directory."""
import json
import os
from pathlib import Path
import threading
import unittest


@unittest.skipUnless(os.getenv("YEGOU_GATE_TEST") == "1", "requires isolated workbench container environment")
class ServiceGateTests(unittest.TestCase):
    def setUp(self):
        from app import main
        from fastapi.testclient import TestClient
        self.main = main
        main.SERVICE_DRAINING = False
        main.SERVICE_REQUESTS = 0
        main.SERVICE_LEASE = ""
        main.SERVICE_LEASE_UNTIL = 0
        self.client = TestClient(main.app)

    def test_drain_refuses_new_work_but_health_and_resume_work(self):
        self.assertTrue(self.client.post("/api/system/drain").json()["idle"])
        self.assertEqual(self.client.post("/api/tts/generate", json={}).status_code, 503)
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        self.client.post("/api/system/undrain")
        self.assertEqual(self.client.post("/api/tts/generate", json={}).status_code, 422)

    def test_lease_excludes_other_writers_and_prevents_stop(self):
        token = self.client.post("/api/system/lease").json()["token"]
        self.assertEqual(self.client.post("/api/system/lease").status_code, 409)
        self.assertEqual(self.client.post("/api/tts/generate", json={}).status_code, 409)
        self.assertFalse(self.client.post("/api/system/drain").json()["idle"])
        self.assertEqual(self.client.post("/api/tts/generate", json={}, headers={"X-Yegou-Lease": token}).status_code, 422)
        self.assertEqual(self.client.post("/api/system/lease/release").status_code, 409)
        self.client.post("/api/system/lease/release", headers={"X-Yegou-Lease": token})
        self.assertTrue(self.client.post("/api/system/drain").json()["idle"])

    def test_live_media_worker_prevents_shutdown(self):
        finish = threading.Event()
        thread = threading.Thread(target=finish.wait, name="dubbing-generation-test")
        thread.start()
        try:
            self.assertFalse(self.client.post("/api/system/drain").json()["idle"])
            self.assertEqual(self.client.post("/api/system/lease").status_code, 409)
        finally:
            finish.set()
            thread.join()

    def test_recovery_marks_abandoned_jobs_failed(self):
        path = self.main.DUBBING_JOBS / "test-recovery.json"
        path.write_text(json.dumps({"status": "generating"}))
        try:
            self.main.recover_interrupted_media_jobs()
            self.assertEqual(json.loads(path.read_text())["status"], "failed")
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
