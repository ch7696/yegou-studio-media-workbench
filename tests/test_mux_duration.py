"""Real FFmpeg regression checks inside the workbench image."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.getenv("YEGOU_GATE_TEST") == "1", "requires FFmpeg and workbench packages")
class MuxTests(unittest.TestCase):
    def test_short_new_audio_keeps_video_tail(self):
        from app.dubbing import _mux_video
        from app.utils import probe
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, audio, target = root / "source.mp4", root / "new.wav", root / "result.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=25",
                            "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=24000", "-t", "3",
                            "-c:v", "libx264", "-c:a", "aac", str(source)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=700:sample_rate=24000",
                            "-t", "0.5", "-c:a", "pcm_s16le", str(audio)], check=True)
            self.assertEqual(_mux_video(source, audio, target), target)
            info = probe(target)
            self.assertAlmostEqual(float(info["format"]["duration"]), 3, delta=.1)
            streams = {s["codec_type"]: s for s in info["streams"]}
            self.assertEqual(streams["video"]["nb_frames"], "75")
            self.assertGreater(float(streams["audio"]["duration"]), 2.9)

    def test_audio_without_picture_is_not_exported_as_video(self):
        from app.dubbing import _mux_video
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "audio.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                            "-t", "0.1", str(audio)], check=True)
            self.assertIsNone(_mux_video(audio, audio, root / "result.mp4"))


if __name__ == "__main__":
    unittest.main()
