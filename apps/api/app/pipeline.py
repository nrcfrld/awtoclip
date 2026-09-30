import json
import os
import subprocess
import time
import threading
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
from fastapi import HTTPException

from .storage import atomic_json, project, project_dir, read_json, save_project, source_path

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CATEGORIES = {"educational", "question", "breakthrough", "testimonial", "promotional", "funny", "other"}
SUMMARY_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}, "topics": {"type": "array", "items": {"type": "string"}}, "participants_context": {"type": "string"}}, "required": ["summary", "topics", "participants_context"], "additionalProperties": False}
CONTEXT_SCHEMA = {"type": "object", "properties": {"session_type": {"type": "string"}, "session_summary": {"type": "string"}, "main_topics": {"type": "array", "items": {"type": "string"}}, "participants_context": {"type": "string"}}, "required": ["session_type", "session_summary", "main_topics", "participants_context"], "additionalProperties": False}
MOMENT_ITEM_SCHEMA = {"type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"}, "title": {"type": "string"}, "category": {"type": "string", "enum": sorted(CATEGORIES)}, "score": {"type": "integer"}, "reason": {"type": "string"}}, "required": ["start", "end", "title", "category", "score", "reason"], "additionalProperties": False}
MOMENTS_SCHEMA = {"type": "object", "properties": {"moments": {"type": "array", "items": MOMENT_ITEM_SCHEMA}}, "required": ["moments"], "additionalProperties": False}
BOUNDARY_SCHEMA = {"type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"}}, "required": ["start", "end"], "additionalProperties": False}
ANALYSIS_CHUNK_SECONDS = 1200
ANALYSIS_OVERLAP_SECONDS = 120
ANALYSIS_LOOKAHEAD_SECONDS = 120
OPENROUTER_LOCK = threading.Lock()
ANALYSIS_RETRY_COOLDOWN_SECONDS = 10


class OpenRouterRateLimit(RuntimeError):
    pass


class OpenRouterModelsExhausted(OpenRouterRateLimit):
    def __init__(self, failures: list[str], retry_after: float = 0):
        super().__init__("All configured free OpenRouter models failed: " + "; ".join(failures))
        self.retry_after = retry_after


def free_models() -> list[str]:
    value = os.getenv("OPENROUTER_MODELS", "")
    models = [model.strip() for model in value.split(",") if model.strip()]
    if not models:
        raise RuntimeError("Set OPENROUTER_MODELS to comma-separated free model IDs in .env")
    if any(not (model.endswith(":free") or model == "openrouter/free") for model in models):
        raise RuntimeError("OPENROUTER_MODELS may contain only :free models or openrouter/free")
    return models


def retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0


def response_error(response) -> str:
    try:
        error = response.json().get("error", {})
        detail = error.get("metadata", {}).get("raw") or error.get("message", response.text)
    except (AttributeError, TypeError, ValueError):
        detail = response.text
    return str(detail)[:500]


def provider_unavailable(status: int, detail: str) -> bool:
    if status in {408, 429, 500, 502, 503, 504, 529}:
        return True
    lowered = detail.lower()
    return any(term in lowered for term in ("provider unavailable", "no providers available", "provider is unavailable"))


def set_state(project_id: str, state: str, step: str, progress: int | None = None, error: str | None = None):
    data = project(project_id)
    data["state"] = state
    data["progress"] = progress
    data["error"] = error
    data.setdefault("stages", {})[step] = "failed" if error else "completed" if state == "READY" else "running"
    save_project(project_id, data)


def complete_stage(project_id: str, step: str):
    data = project(project_id)
    data["stages"][step] = "completed"
    if data.get("state") != "ANALYSIS_PAUSED":
        data["state"] = "READY" if data["stages"].get("analysis") == "completed" else "CREATED"
    data["progress"] = 100
    data["error"] = None
    save_project(project_id, data)


def run(*args: str) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:] or f"Command failed: {args[0]}")


def probe(project_id: str) -> dict:
    video = source_path(project_id)
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], capture_output=True, text=True)
    if result.returncode:
        raise HTTPException(422, "ffprobe could not read this video")
    info = json.loads(result.stdout)
    stream = next((item for item in info.get("streams", []) if item.get("codec_type") == "video"), None)
    if not stream:
        raise HTTPException(422, "No video stream found")
    rate = stream.get("avg_frame_rate", "0/1").split("/")
    fps = round(float(rate[0]) / float(rate[1]), 3) if len(rate) == 2 and float(rate[1]) else 0
    return {"duration": float(info["format"].get("duration", 0)), "width": stream.get("width"), "height": stream.get("height"), "fps": fps, "codec": stream.get("codec_name"), "file_size": video.stat().st_size}


def extract_audio(project_id: str, force: bool = False) -> Path:
    folder = project_dir(project_id)
    audio = folder / "audio.wav"
    if audio.exists() and valid_audio(audio) and not force:
        complete_stage(project_id, "audio")
        return audio
    set_state(project_id, "AUDIO_EXTRACTING", "audio", 0)
    temp = folder / "audio.partial.wav"
    try:
        run("ffmpeg", "-nostdin", "-y", "-i", str(source_path(project_id)), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temp))
        temp.replace(audio)
    finally:
        temp.unlink(missing_ok=True)
    complete_stage(project_id, "audio")
    return audio


def valid_audio(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnchannels() == 1 and handle.getframerate() == 16000 and handle.getsampwidth() == 2 and handle.getnframes() > 0
    except (OSError, EOFError, wave.Error):
        return False


def transcribe(project_id: str, force: bool = False) -> dict:
    path = project_dir(project_id) / "transcript.json"
    if path.exists() and not force:
        result = read_json(path)
        if isinstance(result.get("segments"), list) and result["segments"]:
            complete_stage(project_id, "transcription")
            return result
    replacing = path.exists()
    extract_audio(project_id)
    if replacing:
        folder = project_dir(project_id)
        for name in ("context-chunks.json", "global-context.json", "analysis-chunks.json", "analysis-refinements.json"):
            (folder / name).unlink(missing_ok=True)
        moments_path = folder / "moments.json"
        if moments_path.exists():
            manual = [item for item in read_json(moments_path).get("moments", []) if item.get("source") == "manual"]
            if manual:
                atomic_json(moments_path, {"moments": manual})
            else:
                moments_path.unlink()
        data = project(project_id)
        data["stages"]["global_context"] = "pending"
        data["stages"]["analysis"] = "pending"
        save_project(project_id, data)
    set_state(project_id, "TRANSCRIBING", "transcription", 0)
    url = os.getenv("WHISPER_SERVICE_URL", "http://host.docker.internal:9001").rstrip("/")
    with httpx.Client(timeout=None) as client:
        response = client.post(f"{url}/transcribe", json={"project_id": project_id, "force": force})
        response.raise_for_status()
        result = response.json()
    if not isinstance(result.get("segments"), list) or not result["segments"]:
        raise RuntimeError("Whisper returned an empty transcript")
    atomic_json(path, result)
    complete_stage(project_id, "transcription")
    return result


def ask_llm(system: str, payload: dict, schema: dict | None = None, schema_name: str = "result") -> dict:
    result, _model = ask_llm_with_model(system, payload, schema, schema_name)
    return result


def ask_llm_with_model(system: str, payload: dict, schema: dict | None = None, schema_name: str = "result") -> tuple[dict, str]:
    with OPENROUTER_LOCK:
        return _ask_llm_unlocked(system, payload, schema, schema_name)


def _ask_llm_unlocked(system: str, payload: dict, schema: dict | None = None, schema_name: str = "result") -> tuple[dict, str]:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env")
    models = free_models()
    response_format = {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}} if schema else {"type": "json_object"}
    base_request = {"temperature": 0.2, "max_tokens": 8192, "response_format": response_format, "messages": [{"role": "system", "content": system + " Return a JSON object only."}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
    failures = []
    retry_after = 0.0
    with httpx.Client(timeout=180) as client:
        for model in models:
            request = {**base_request, "model": model}
            if model != "openrouter/free":
                request["reasoning"] = {"enabled": False}
            reasoning_fallback_used = False
            while True:
                try:
                    response = client.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=request)
                except (httpx.TimeoutException, httpx.ConnectError) as exc:
                    failures.append(f"{model}: {type(exc).__name__}")
                    break
                if response.is_error:
                    detail = response_error(response)
                    if response.status_code == 400 and "reasoning" in request and not reasoning_fallback_used and "reasoning" in detail.lower():
                        request.pop("reasoning")
                        reasoning_fallback_used = True
                        continue
                    if provider_unavailable(response.status_code, detail):
                        retry_after = max(retry_after, retry_after_seconds(response.headers.get("Retry-After")))
                        failures.append(f"{model}: OpenRouter {response.status_code}: {detail}")
                        break
                    raise RuntimeError(f"OpenRouter {response.status_code}: {detail}")
                try:
                    content = response.json()["choices"][0]["message"].get("content") or ""
                    if content.startswith("```"):
                        content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                    return json.loads(content), model
                except (KeyError, IndexError, TypeError, ValueError):
                    failures.append(f"{model}: invalid JSON response")
                    break
    raise OpenRouterModelsExhausted(failures, retry_after)


def transcript_chunks(segments: list[dict], target: float = ANALYSIS_CHUNK_SECONDS, overlap: float = ANALYSIS_OVERLAP_SECONDS) -> list[list[dict]]:
    if not segments:
        return []
    chunks = []
    start = float(segments[0]["start"])
    last_end = float(segments[-1]["end"])
    while start < last_end:
        end = start + target
        chunk = [s for s in segments if float(s["end"]) > start and float(s["start"]) < end]
        if chunk:
            chunks.append(chunk)
        if end >= last_end:
            break
        start = end - overlap
    return chunks


def build_context(project_id: str, transcript: dict) -> dict:
    path = project_dir(project_id) / "global-context.json"
    if path.exists():
        result = read_json(path)
        if result.get("session_summary"):
            complete_stage(project_id, "global_context")
            return result
    set_state(project_id, "BUILDING_CONTEXT", "global_context", 0)
    segments = transcript["segments"]
    summaries_path = project_dir(project_id) / "context-chunks.json"
    summaries = read_json(summaries_path).get("summaries", []) if summaries_path.exists() else []
    analysis_path = project_dir(project_id) / "analysis-chunks.json"
    analysis_records = read_json(analysis_path).get("chunks", {}) if analysis_path.exists() else {}
    if analysis_records and all(record.get("status") == "completed" and record.get("summary") for record in analysis_records.values()):
        summaries = [
            {"start": record["start"], "end": record["end"], **record["summary"]}
            for _, record in sorted(analysis_records.items(), key=lambda item: int(item[0]))
        ]
        atomic_json(summaries_path, {"summaries": summaries})
    else:
        chunks = transcript_chunks(segments, target=ANALYSIS_CHUNK_SECONDS, overlap=0)
        for index in range(len(summaries), len(chunks)):
            chunk = chunks[index]
            response = ask_llm("Summarize this portion of a recording. Return {\"summary\": string, \"topics\": string[], \"participants_context\": string}. Capture all important topics briefly.", {"start": chunk[0]["start"], "end": chunk[-1]["end"], "segments": chunk}, SUMMARY_SCHEMA, "chunk_summary")
            summaries.append({"start": chunk[0]["start"], "end": chunk[-1]["end"], "summary": response.get("summary", ""), "topics": response.get("topics", []), "participants_context": response.get("participants_context", "")})
            atomic_json(summaries_path, {"summaries": summaries})
            set_state(project_id, "BUILDING_CONTEXT", "global_context", int(90 * len(summaries) / max(1, len(chunks))))
    result = ask_llm("Combine summaries covering the entire recording into compact global context. Provide session_type, session_summary, main_topics (array), participants_context. Do not copy the full transcript.", {"chunk_summaries": summaries}, CONTEXT_SCHEMA, "global_context")
    context = {"session_type": str(result.get("session_type", "recording")), "session_summary": str(result.get("session_summary", "")), "main_topics": result.get("main_topics", []), "participants_context": str(result.get("participants_context", ""))}
    atomic_json(path, context)
    complete_stage(project_id, "global_context")
    return context


def excerpt(segments: list[dict], start: float, end: float) -> str:
    return " ".join(str(s["text"]).strip() for s in segments if float(s["end"]) > start and float(s["start"]) < end)[:1200]


def normalize(raw: dict, segments: list[dict], duration: float, max_duration: float) -> dict | None:
    try:
        start, end = float(raw["start"]), float(raw["end"])
        if start < 0 or end <= start or start >= duration:
            return None
        end = min(end, duration, start + max_duration)
        score = max(0, min(100, int(raw.get("score", 50))))
    except (TypeError, ValueError, KeyError):
        return None
    category = str(raw.get("category", "other")).lower()
    return {"id": uuid.uuid4().hex[:12], "start": round(start, 3), "end": round(end, 3), "title": str(raw.get("title", "Untitled moment"))[:120], "category": category if category in CATEGORIES else "other", "score": score, "reason": str(raw.get("reason", ""))[:1000], "excerpt": excerpt(segments, start, end), "selected": False, "rejected": False, "source": "ai"}


def deduplicate(moments: list[dict]) -> list[dict]:
    result = []
    for moment in sorted(moments, key=lambda m: m["score"], reverse=True):
        duplicate = any(max(0, min(moment["end"], prior["end"]) - max(moment["start"], prior["start"])) / max(0.001, min(moment["end"] - moment["start"], prior["end"] - prior["start"])) > 0.65 for prior in result)
        if not duplicate:
            result.append(moment)
    return result


def moment_items(response: object) -> list[dict]:
    if isinstance(response, dict):
        for container in (response, response.get("result"), response.get("data")):
            if isinstance(container, dict):
                for key in ("moments", "candidates", "clips", "highlights"):
                    value = container.get(key)
                    if isinstance(value, list):
                        return [item for item in value if isinstance(item, dict)]
        if "start" in response and "end" in response:
            return [response]
        raise RuntimeError(f"OpenRouter returned no moments array (keys: {', '.join(map(str, list(response)[:8]))})")
    raise RuntimeError("OpenRouter returned an invalid moments response")


def partial_segments(project_id: str) -> list[dict]:
    path = project_dir(project_id) / "transcript.partial.jsonl"
    if not path.exists():
        return []
    result = []
    with path.open() as source:
        for line in source:
            try:
                item = json.loads(line)
            except ValueError:
                break
            if isinstance(item, dict) and {"start", "end", "text"} <= item.keys():
                result.append(item)
    return result


def chunk_windows(total: float) -> list[tuple[int, float, float]]:
    windows = []
    start = 0.0
    index = 0
    while start < total:
        windows.append((index, start, min(total, start + ANALYSIS_CHUNK_SECONDS)))
        if start + ANALYSIS_CHUNK_SECONDS >= total:
            break
        start += ANALYSIS_CHUNK_SECONDS - ANALYSIS_OVERLAP_SECONDS
        index += 1
    return windows


def analyze_chunk(project_id: str, segments: list[dict], index: int, start: float, end: float, records: dict, path: Path) -> None:
    key = str(index)
    record = records.setdefault(key, {"start": start, "end": end})
    record["status"] = "running"
    atomic_json(path, {"chunks": records})
    chunk = [segment for segment in segments if float(segment["end"]) > start and float(segment["start"]) < end]
    if not chunk:
        record.update(status="completed", candidates=[], summary=None)
        atomic_json(path, {"chunks": records})
        return
    if not record.get("summary"):
        summary, model = ask_llm_with_model(
            "Summarize this portion of a recording. Return summary, topics, and participants_context. Keep it compact.",
            {"start": start, "end": end, "segments": chunk},
            SUMMARY_SCHEMA,
            "chunk_summary",
        )
        record["summary"] = summary
        record["summary_model"] = model
        atomic_json(path, {"chunks": records})
    prior = [records[str(i)]["summary"] for i in range(index) if str(i) in records and records[str(i)].get("summary")]
    context = {
        "session_type": "recording",
        "session_summary": " ".join(str(item.get("summary", "")) for item in prior + [record["summary"]])[-3000:],
        "main_topics": list(dict.fromkeys(topic for item in prior + [record["summary"]] for topic in item.get("topics", [])))[:30],
        "participants_context": str(record["summary"].get("participants_context", "")),
    }
    response, model = ask_llm_with_model(
        "Find ALL useful standalone clip moments in this transcript chunk. Categories: educational, question, breakthrough, testimonial, promotional, funny, other. Include start, end, title, category, score 0-100, reason. Respect max clip duration and original timestamps. Return {\"moments\": [...]}.",
        {"global_context": context, "max_clip_duration": float(project(project_id).get("max_clip_duration", 60)), "chunk_start": start, "chunk_end": end, "segments": chunk},
        MOMENTS_SCHEMA,
        "moments",
    )
    record.update(status="completed", candidates=moment_items(response), model=model, error=None)
    record.pop("retry_at", None)
    atomic_json(path, {"chunks": records})


def analyze(project_id: str, force: bool = False) -> list[dict]:
    folder = project_dir(project_id)
    output = folder / "moments.json"
    if output.exists() and not force and project(project_id).get("stages", {}).get("analysis") == "completed":
        result = read_json(output)
        if isinstance(result.get("moments"), list):
            set_state(project_id, "READY", "analysis", 100)
            return result["moments"]
    chunks_path = folder / "analysis-chunks.json"
    records = read_json(chunks_path).get("chunks", {}) if chunks_path.exists() and not force else {}
    if force:
        (folder / "analysis-refinements.json").unlink(missing_ok=True)
    transcript = None
    paused_error = None
    cooldown_retries = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        transcription = pool.submit(transcribe, project_id)
        while True:
            if transcription.done():
                transcript = transcription.result()
            progress_path = folder / "transcription-progress.json"
            progress = read_json(progress_path) if progress_path.exists() else {}
            total = float(progress.get("total_audio_seconds") or project(project_id)["duration"])
            processed = float(progress.get("processed_audio_seconds") or 0)
            if transcript is not None:
                processed = total
            segments_now = transcript["segments"] if transcript is not None else partial_segments(project_id)
            ready = None
            for index, start, end in chunk_windows(total):
                record = records.get(str(index), {})
                if record.get("status") == "completed":
                    continue
                if transcript is not None or processed >= end + ANALYSIS_LOOKAHEAD_SECONDS:
                    ready = (index, start, end)
                break
            if ready and not paused_error:
                index, start, end = ready
                if time.time() < records.get(str(index), {}).get("retry_at", 0):
                    time.sleep(0.25)
                    continue
                try:
                    if transcript is not None:
                        completed = sum(record.get("status") == "completed" for record in records.values())
                        set_state(project_id, "ANALYZING", "analysis", int(100 * completed / max(1, len(chunk_windows(total)))))
                    analyze_chunk(project_id, segments_now, index, start, end, records, chunks_path)
                except OpenRouterModelsExhausted as exc:
                    retry_at = time.time() + max(ANALYSIS_RETRY_COOLDOWN_SECONDS, exc.retry_after)
                    records[str(index)].update(status="pending", error=str(exc), retry_at=retry_at)
                    atomic_json(chunks_path, {"chunks": records})
                    data = project(project_id)
                    data.update(state="ANALYSIS_PAUSED", error=str(exc), progress=None)
                    data.setdefault("stages", {})["analysis"] = "paused"
                    save_project(project_id, data)
                    if cooldown_retries:
                        paused_error = exc
                    else:
                        cooldown_retries += 1
                continue
            if transcript is not None:
                if paused_error:
                    raise paused_error
                if all(records.get(str(index), {}).get("status") == "completed" for index, _, _ in chunk_windows(total)):
                    break
            time.sleep(0.25)

    context = build_context(project_id, transcript)
    segments = transcript["segments"]
    max_duration = float(project(project_id).get("max_clip_duration", 60))
    duration = float(project(project_id)["duration"])
    candidates = [candidate for index, _, _ in chunk_windows(total) for candidate in records[str(index)].get("candidates", [])]
    set_state(project_id, "REFINING", "analysis", 0)
    refinements_path = folder / "analysis-refinements.json"
    saved_refinements = read_json(refinements_path).get("refined", {}) if refinements_path.exists() else {}
    refined = []
    for index, candidate in enumerate(candidates):
        if str(index) in saved_refinements:
            if saved_refinements[str(index)]:
                refined.append(saved_refinements[str(index)])
            continue
        try:
            start, end = float(candidate["start"]), float(candidate["end"])
        except (KeyError, ValueError, TypeError):
            saved_refinements[str(index)] = None
            atomic_json(refinements_path, {"refined": saved_refinements})
            continue
        nearby = [s for s in segments if float(s["end"]) > start - 60 and float(s["start"]) < end + 60]
        response = ask_llm("Refine this candidate's start and end using the surrounding transcript, including neighboring chunk context. Start at a natural sentence, include needed setup, end at conclusion. Maximum duration is strict. Return {\"start\": number, \"end\": number}.", {"global_context": context, "candidate": candidate, "surrounding_segments": nearby, "max_clip_duration": max_duration}, BOUNDARY_SCHEMA, "boundaries")
        normalized = normalize({**candidate, "start": response.get("start", start), "end": response.get("end", end)}, segments, duration, max_duration)
        if normalized:
            refined.append(normalized)
        saved_refinements[str(index)] = normalized
        atomic_json(refinements_path, {"refined": saved_refinements})
        set_state(project_id, "REFINING", "analysis", int(100 * (index + 1) / max(1, len(candidates))))
    manual_moments = [moment for moment in read_json(output).get("moments", []) if moment.get("source") == "manual"] if output.exists() else []
    final = deduplicate(refined) + manual_moments
    atomic_json(output, {"moments": final})
    set_state(project_id, "READY", "analysis", 100)
    return final


def export_clip(project_id: str, moment: dict, number: int) -> str:
    folder = project_dir(project_id) / "clips"
    folder.mkdir(exist_ok=True)
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", moment["title"].lower()).strip("-")[:50] or "clip"
    name = f"{number:02d}-{slug}.mp4"
    target = folder / name
    temp = folder / f".{name}"
    try:
        run("ffmpeg", "-nostdin", "-y", "-ss", str(moment["start"]), "-i", str(source_path(project_id)), "-t", str(moment["end"] - moment["start"]), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-movflags", "+faststart", "-f", "mp4", str(temp))
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return name
