import json
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import main


class StreamingWhisperTests(unittest.TestCase):
    def make_audio(self, folder: Path, seconds: int = 65) -> Path:
        path = folder / "audio.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0\0" * 16000 * seconds)
        return path

    def test_progress_eta_gate_and_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            early = main.write_progress(path, 5, 100, 5, 2, "en", "running")
            self.assertIsNone(early["estimated_remaining_seconds"])
            self.assertEqual(early["eta_message"], "Estimating remaining time...")
            self.assertEqual(early["realtime_speed"], 1)
            by_time = main.write_progress(path, 5, 100, 30, 2, "en", "running")
            self.assertEqual(by_time["estimated_remaining_seconds"], 47.5)
            by_percent = main.write_progress(path, 10, 100, 5, 2, "en", "running")
            self.assertEqual(by_percent["estimated_remaining_seconds"], 45)

    def test_segments_are_appended_before_completion_and_resume_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = self.make_audio(folder)
            calls = []

            def fake_transcribe(_samples, **_kwargs):
                calls.append(1)
                if len(calls) == 2:
                    partial = main.read_partial(folder / "transcript.partial.jsonl")
                    self.assertEqual(len(partial), 1)
                    self.assertEqual(json.loads((folder / "transcription-progress.json").read_text())["processed_audio_seconds"], 30)
                    raise RuntimeError("temporary failure")
                return {"language": "en", "segments": [{"start": 0, "end": 5, "text": f"Window {len(calls)}", "words": [{"start": 0, "end": 1, "word": "Window"}]}]}

            import mlx_whisper
            from mlx_whisper.transcribe import ModelHolder
            with patch.object(ModelHolder, "get_model", return_value=object()), patch.object(mlx_whisper, "transcribe", side_effect=fake_transcribe):
                with self.assertRaisesRegex(RuntimeError, "temporary failure"):
                    main.transcribe_audio(audio, folder)
                self.assertEqual(json.loads((folder / "transcription-progress.json").read_text())["status"], "failed")
                result = main.transcribe_audio(audio, folder)

            self.assertEqual(len(result["segments"]), 3)
            self.assertEqual([item["start"] for item in result["segments"]], [0, 30, 60])
            self.assertEqual(result["segments"][1]["words"][0]["start"], 30)
            self.assertEqual(len(main.read_partial(folder / "transcript.partial.jsonl")), 3)
            self.assertEqual(json.loads((folder / "transcription-progress.json").read_text())["progress_percentage"], 100)

    def test_model_loading_is_excluded_from_processing_time(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = self.make_audio(folder, 1)
            import mlx_whisper
            from mlx_whisper.transcribe import ModelHolder

            def load_model(*_args):
                progress = json.loads((folder / "transcription-progress.json").read_text())
                self.assertEqual(progress["status"], "loading_model")
                time.sleep(0.15)

            response = {"language": "en", "segments": [{"start": 0, "end": 1, "text": "One second", "words": []}]}
            with patch.object(ModelHolder, "get_model", side_effect=load_model), patch.object(mlx_whisper, "transcribe", return_value=response):
                main.transcribe_audio(audio, folder)
            progress = json.loads((folder / "transcription-progress.json").read_text())
            self.assertLess(progress["processing_elapsed_seconds"], 0.1)


if __name__ == "__main__":
    unittest.main()
