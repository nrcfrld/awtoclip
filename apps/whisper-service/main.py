import json
import os
import re
import time
import wave
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Native macOS Whisper service")
PROJECT_ID = re.compile(r"^[a-f0-9]{12}$")
HOST_DATA_ROOT = Path(os.getenv("HOST_DATA_ROOT", "data")).resolve()
MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-small-mlx")
WINDOW_SECONDS = 30
EMA_WEIGHT = 0.3


class Request(BaseModel):
    project_id: str
    force: bool = False


def project_folder(project_id: str) -> Path:
    if not PROJECT_ID.fullmatch(project_id):
        raise HTTPException(422, "Invalid project ID")
    return HOST_DATA_ROOT / "projects" / project_id


def read_partial(path: Path) -> list[dict]:
    if not path.exists():
        return []
    segments = []
    valid_lines = []
    for line in path.read_text().splitlines():
        try:
            segment = json.loads(line)
        except ValueError:
            break
        if isinstance(segment, dict) and {"start", "end", "text"} <= segment.keys():
            segments.append(segment)
            valid_lines.append(line)
    if len(valid_lines) != len(path.read_text().splitlines()):
        path.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""))
    return segments


def write_progress(path: Path, processed: float, total: float, elapsed: float, speed: float | None, language: str | None, status: str) -> dict:
    percentage = round(100 * processed / total, 2) if total else 0.0
    stable = elapsed >= 30 or percentage >= 10
    value = {
        "processed_audio_seconds": round(processed, 3),
        "total_audio_seconds": round(total, 3),
        "progress_percentage": min(100.0, percentage),
        "realtime_speed": round(processed / elapsed, 3) if elapsed > 0 else None,
        "estimated_remaining_seconds": round((total - processed) / speed, 1) if stable and speed and processed < total else (0.0 if processed >= total else None),
        "eta_message": None if stable or processed >= total else "Estimating remaining time...",
        "smoothed_speed": round(speed, 3) if speed else None,
        "processing_elapsed_seconds": round(elapsed, 3),
        "language": language,
        "status": status,
    }
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value))
    temp.replace(path)
    return value


def transcribe_audio(audio: Path, folder: Path, force: bool = False) -> dict:
    import mlx.core as mx
    import mlx_whisper
    import numpy as np
    from mlx_whisper.transcribe import ModelHolder

    output = folder / "transcript.json"
    partial = folder / "transcript.partial.jsonl"
    progress_path = folder / "transcription-progress.json"
    if output.is_file() and not force:
        try:
            cached = json.loads(output.read_text())
            if cached.get("segments"):
                return cached
        except (OSError, ValueError):
            pass
    if force:
        partial.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    with wave.open(str(audio), "rb") as source:
        if (source.getnchannels(), source.getframerate(), source.getsampwidth()) != (1, 16000, 2):
            raise ValueError("Expected 16 kHz mono 16-bit WAV")
        total = source.getnframes() / source.getframerate()
        previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        processed = min(total, float(previous.get("processed_audio_seconds", 0)))
        elapsed_before = float(previous.get("processing_elapsed_seconds", 0))
        language = previous.get("language")
        speed = previous.get("smoothed_speed")
        segments = read_partial(partial)
        write_progress(progress_path, processed, total, elapsed_before, speed, language, "loading_model")
        # Model loading is deliberately outside the processing clock.
        try:
            ModelHolder.get_model(MODEL, mx.float16)
        except Exception:
            write_progress(progress_path, processed, total, elapsed_before, speed, language, "failed")
            raise
        started = time.monotonic()
        write_progress(progress_path, processed, total, elapsed_before, speed, language, "running")
        try:
            while processed < total:
                source.setpos(round(processed * source.getframerate()))
                frame_count = min(round(WINDOW_SECONDS * source.getframerate()), source.getnframes() - source.tell())
                samples = np.frombuffer(source.readframes(frame_count), dtype=np.int16).astype(np.float32) / 32768.0
                window_started = time.monotonic()
                result = mlx_whisper.transcribe(samples, path_or_hf_repo=MODEL, word_timestamps=True, verbose=None, language=language, initial_prompt=" ".join(s["text"] for s in segments[-8:])[-224:] or None)
                language = result.get("language") or language
                window_end = min(total, processed + frame_count / source.getframerate())
                existing_end = max((float(s["end"]) for s in segments), default=0.0)
                new_segments = []
                for item in result.get("segments", []):
                    words = [{**word, "start": round(processed + float(word["start"]), 3), "end": round(processed + float(word["end"]), 3)} for word in item.get("words", [])]
                    segment = {"start": round(processed + float(item["start"]), 3), "end": round(processed + float(item["end"]), 3), "text": str(item["text"]).strip(), "words": words}
                    if segment["text"] and segment["end"] > existing_end and segment["end"] > segment["start"]:
                        new_segments.append(segment)
                with partial.open("a") as handle:
                    for segment in new_segments:
                        handle.write(json.dumps(segment, ensure_ascii=False) + "\n")
                segments.extend(new_segments)
                window_elapsed = max(0.001, time.monotonic() - window_started)
                instant_speed = (window_end - processed) / window_elapsed
                speed = instant_speed if speed is None else EMA_WEIGHT * instant_speed + (1 - EMA_WEIGHT) * speed
                processed = window_end
                write_progress(progress_path, processed, total, elapsed_before + time.monotonic() - started, speed, language, "running")
        except Exception:
            write_progress(progress_path, processed, total, elapsed_before + time.monotonic() - started, speed, language, "failed")
            raise

    if not segments:
        write_progress(progress_path, processed, total, elapsed_before + time.monotonic() - started, speed, language, "failed")
        raise ValueError("Whisper returned no segments")
    payload = {"language": language, "segments": segments}
    temp = folder / "transcript.partial.json"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temp.replace(output)
    write_progress(progress_path, total, total, elapsed_before + time.monotonic() - started, speed, language, "completed")
    return payload


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}


@app.get("/projects/{project_id}/progress")
def progress(project_id: str):
    path = project_folder(project_id) / "transcription-progress.json"
    if not path.is_file():
        raise HTTPException(404, "No transcription progress yet")
    return json.loads(path.read_text())


@app.post("/transcribe")
def transcribe(body: Request):
    folder = project_folder(body.project_id)
    audio = folder / "audio.wav"
    if not audio.is_file():
        raise HTTPException(404, "Project audio not found")
    try:
        return transcribe_audio(audio, folder, body.force)
    except Exception as exc:
        raise HTTPException(500, f"Whisper transcription failed: {exc}") from exc
