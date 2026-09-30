import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import pipeline
from .storage import PROJECTS, atomic_json, project, project_dir, read_json, save_project, source_path

@asynccontextmanager
async def lifespan(_app: FastAPI):
    interrupted = {"UPLOADING", "AUDIO_EXTRACTING", "TRANSCRIBING", "BUILDING_CONTEXT", "ANALYZING", "REFINING", "EXPORTING"}
    for path in PROJECTS.glob("*/project.json"):
        try:
            data = read_json(path)
            if data.get("state") in interrupted:
                data.update(state="FAILED", error="Processing was interrupted. Retry to resume from saved stages.", progress=None)
                for name, status in data.get("stages", {}).items():
                    if status == "running":
                        data["stages"][name] = "failed"
                atomic_json(path, data)
        except HTTPException:
            continue
    yield


app = FastAPI(title="Local AI Video Clipper", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
tasks: dict[str, asyncio.Task] = {}


class CreateProject(BaseModel):
    max_clip_duration: float = Field(default=60, ge=5, le=600)


class MomentPatch(BaseModel):
    start: float | None = None
    end: float | None = None
    title: str | None = None
    category: str | None = None
    selected: bool | None = None
    rejected: bool | None = None


class ManualMoment(BaseModel):
    start: float
    end: float
    title: str = "Manual clip"
    category: str = "other"


class ClipRequest(BaseModel):
    moment_ids: list[str] | None = None


def validate_times(start: float, end: float, data: dict):
    if not (0 <= start < end <= data.get("duration", 0)):
        raise HTTPException(422, "Clip must fit inside the source video")
    if end - start > data["max_clip_duration"] + 0.001:
        raise HTTPException(422, "Clip exceeds maximum duration")


def update_moments(project_id: str, moments: list[dict]):
    atomic_json(project_dir(project_id) / "moments.json", {"moments": moments})


def project_view(project_id: str) -> dict:
    current = project(project_id)
    folder = project_dir(project_id)
    progress_path = folder / "transcription-progress.json"
    if progress_path.exists():
        try:
            current["transcription_progress"] = read_json(progress_path)
        except HTTPException:
            pass
    chunks_path = folder / "analysis-chunks.json"
    if chunks_path.exists():
        try:
            records = read_json(chunks_path).get("chunks", {})
            current["analysis_progress"] = {
                "completed_chunks": sum(item.get("status") == "completed" for item in records.values()),
                "paused": any(item.get("status") in {"paused", "pending"} and item.get("error") for item in records.values()),
            }
        except HTTPException:
            pass
    return current


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/projects")
def list_projects():
    found = []
    for path in PROJECTS.glob("*/project.json"):
        try:
            found.append(read_json(path))
        except HTTPException:
            continue
    return sorted(found, key=lambda p: p.get("created_at", ""), reverse=True)


@app.post("/projects")
def create_project(body: CreateProject):
    from datetime import datetime, timezone
    project_id = uuid.uuid4().hex[:12]
    folder = PROJECTS / project_id
    folder.mkdir(parents=True)
    value = {"id": project_id, "created_at": datetime.now(timezone.utc).isoformat(), "original_filename": None, "duration": None, "max_clip_duration": body.max_clip_duration, "state": "CREATED", "progress": None, "error": None, "stages": {"upload": "pending", "audio": "pending", "transcription": "pending", "global_context": "pending", "analysis": "pending"}}
    atomic_json(folder / "project.json", value)
    return value


@app.get("/projects/{project_id}")
def get_project(project_id: str):
    return project_view(project_id)


@app.post("/projects/{project_id}/video")
async def upload_video(project_id: str, video: UploadFile = File(...)):
    folder = project_dir(project_id)
    data = project(project_id)
    if data.get("original_filename"):
        raise HTTPException(409, "Project already has a source video")
    filename = Path(video.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv"}:
        raise HTTPException(422, "Use an MP4, MOV, or MKV file")
    data.update(state="UPLOADING", progress=0, error=None)
    save_project(project_id, data)
    temp = folder / f"upload.partial{suffix}"
    target = folder / f"source{suffix}"
    try:
        with temp.open("wb") as out:
            while chunk := await video.read(4 * 1024 * 1024):
                out.write(chunk)
        temp.replace(target)
        metadata = await asyncio.to_thread(pipeline.probe, project_id)
        if metadata["duration"] <= 0:
            raise HTTPException(422, "Video has no usable duration")
        data = project(project_id)
        data.update(metadata)
        data.update(original_filename=filename, state="CREATED", progress=100, error=None)
        data["stages"]["upload"] = "completed"
        save_project(project_id, data)
        return data
    except Exception as exc:
        target.unlink(missing_ok=True)
        data = project(project_id)
        data.update(state="FAILED", error=str(exc), progress=None)
        data["stages"]["upload"] = "failed"
        save_project(project_id, data)
        raise
    finally:
        temp.unlink(missing_ok=True)
        await video.close()


@app.get("/projects/{project_id}/video")
def stream_video(project_id: str, request: Request):
    path = source_path(project_id)
    size = path.stat().st_size
    media = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska"}[path.suffix]
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(path, media_type=media, headers={"Accept-Ranges": "bytes"})
    if not range_header.startswith("bytes=") or "," in range_header:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    try:
        start_text, end_text = range_header[6:].split("-", 1)
        if start_text:
            start = int(start_text)
            end = min(int(end_text), size - 1) if end_text else size - 1
        else:
            length = int(end_text)
            start, end = max(0, size - length), size - 1
        if start < 0 or start >= size or end < start:
            raise ValueError
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def chunks():
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block

    return StreamingResponse(chunks(), status_code=206, media_type=media, headers={"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1)})


async def start_job(project_id: str, name: str, operation):
    source_path(project_id)
    old = tasks.get(project_id)
    if old and not old.done():
        raise HTTPException(409, "Project is already processing")

    async def worker():
        try:
            await asyncio.to_thread(operation)
        except Exception as exc:
            data = project(project_id)
            paused = isinstance(exc, pipeline.OpenRouterRateLimit)
            data.update(state="ANALYSIS_PAUSED" if paused else "FAILED", error=str(exc), progress=None)
            for stage_name, status in data.get("stages", {}).items():
                if status == "running":
                    data["stages"][stage_name] = "failed"
            if paused:
                data["stages"]["analysis"] = "paused"
            save_project(project_id, data)

    tasks[project_id] = asyncio.create_task(worker())
    return {"accepted": True, "operation": name}


@app.post("/projects/{project_id}/extract-audio", status_code=202)
async def extract_audio(project_id: str):
    return await start_job(project_id, "extract-audio", lambda: pipeline.extract_audio(project_id))


@app.post("/projects/{project_id}/transcribe", status_code=202)
async def transcribe(project_id: str, force: bool = False):
    return await start_job(project_id, "transcribe", lambda: pipeline.transcribe(project_id, force=force))


@app.post("/projects/{project_id}/analyze", status_code=202)
async def analyze(project_id: str):
    return await start_job(project_id, "analyze", lambda: pipeline.analyze(project_id))


@app.post("/projects/{project_id}/process", status_code=202)
async def process(project_id: str):
    return await start_job(project_id, "process", lambda: pipeline.analyze(project_id))


@app.get("/projects/{project_id}/transcript")
def transcript(project_id: str):
    return read_json(project_dir(project_id) / "transcript.json")


@app.get("/projects/{project_id}/moments")
def moments(project_id: str):
    return read_json(project_dir(project_id) / "moments.json")


@app.post("/projects/{project_id}/moments")
def create_moment(project_id: str, body: ManualMoment):
    data = project(project_id)
    validate_times(body.start, body.end, data)
    path = project_dir(project_id) / "moments.json"
    existing = read_json(path)["moments"] if path.exists() else []
    transcript_path = project_dir(project_id) / "transcript.json"
    segments = read_json(transcript_path).get("segments", []) if transcript_path.exists() else []
    new = {"id": uuid.uuid4().hex[:12], "start": body.start, "end": body.end, "title": body.title[:120], "category": body.category if body.category in pipeline.CATEGORIES else "other", "score": 0, "reason": "Created manually", "excerpt": pipeline.excerpt(segments, body.start, body.end), "selected": True, "rejected": False, "source": "manual"}
    existing.append(new)
    update_moments(project_id, existing)
    return new


@app.patch("/projects/{project_id}/moments/{moment_id}")
def patch_moment(project_id: str, moment_id: str, body: MomentPatch):
    path = project_dir(project_id) / "moments.json"
    existing = read_json(path)["moments"]
    item = next((m for m in existing if m["id"] == moment_id), None)
    if item is None:
        raise HTTPException(404, "Moment not found")
    changes = body.model_dump(exclude_unset=True)
    start = changes.get("start", item["start"])
    end = changes.get("end", item["end"])
    validate_times(start, end, project(project_id))
    if "category" in changes and changes["category"] not in pipeline.CATEGORIES:
        raise HTTPException(422, "Invalid category")
    item.update(changes)
    if "start" in changes or "end" in changes:
        transcript_path = project_dir(project_id) / "transcript.json"
        if transcript_path.exists():
            item["excerpt"] = pipeline.excerpt(read_json(transcript_path).get("segments", []), start, end)
    update_moments(project_id, existing)
    return item


@app.post("/projects/{project_id}/clips")
async def create_clips(project_id: str, body: ClipRequest):
    existing = read_json(project_dir(project_id) / "moments.json")["moments"]
    chosen = [m for m in existing if m["selected"] and not m["rejected"] and (m["id"] in body.moment_ids if body.moment_ids is not None else True)]
    if not chosen:
        raise HTTPException(422, "Select at least one moment")
    data = project(project_id)
    data.update(state="EXPORTING", progress=0, error=None)
    save_project(project_id, data)
    names = []
    try:
        for index, item in enumerate(chosen):
            names.append(await asyncio.to_thread(pipeline.export_clip, project_id, item, index + 1))
            data = project(project_id)
            data["progress"] = int(100 * (index + 1) / len(chosen))
            save_project(project_id, data)
    except Exception as exc:
        data = project(project_id)
        data.update(state="FAILED", error=str(exc))
        save_project(project_id, data)
        raise HTTPException(500, str(exc)) from exc
    data = project(project_id)
    data.update(state="READY", progress=100)
    save_project(project_id, data)
    return {"clips": [{"filename": name, "url": f"/projects/{project_id}/clips/{name}"} for name in names]}


@app.get("/projects/{project_id}/clips/{filename}")
def get_clip(project_id: str, filename: str):
    if Path(filename).name != filename or not filename.endswith(".mp4"):
        raise HTTPException(404, "Clip not found")
    path = project_dir(project_id) / "clips" / filename
    if not path.is_file():
        raise HTTPException(404, "Clip not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.get("/projects/{project_id}/events")
async def events(project_id: str, request: Request):
    project(project_id)

    async def updates():
        previous = None
        last_id = request.headers.get("last-event-id", "")
        segment_cursor = int(last_id.removeprefix("segment-")) if last_id.startswith("segment-") and last_id[8:].isdigit() else 0
        while True:
            current = project_view(project_id)
            folder = project_dir(project_id)
            encoded = json.dumps(current)
            if encoded != previous:
                yield f"data: {encoded}\n\n"
                previous = encoded
            else:
                yield ": keepalive\n\n"
            partial = folder / "transcript.partial.jsonl"
            if partial.exists():
                with partial.open() as source:
                    for number, line in enumerate(source, 1):
                        if number <= segment_cursor:
                            continue
                        try:
                            json.loads(line)
                        except ValueError:
                            break
                        yield f"event: transcript_segment\nid: segment-{number}\ndata: {line.rstrip()}\n\n"
                        segment_cursor = number
            await asyncio.sleep(1)

    return StreamingResponse(updates(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
