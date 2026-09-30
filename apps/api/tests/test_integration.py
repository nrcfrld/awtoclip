import json
import asyncio
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main, pipeline, storage


def make_video(path: Path):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-t", "4", "-c:v", "libx264", "-c:a", "aac", str(path)], check=True)


def test_video_range_audio_moments_and_export(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr(storage, "PROJECTS", projects)
    monkeypatch.setattr(main, "PROJECTS", projects)
    source = tmp_path / "sample.mp4"
    make_video(source)
    with TestClient(main.app) as client:
        created = client.post("/projects", json={"max_clip_duration": 60}).json()
        project_id = created["id"]
        with source.open("rb") as handle:
            upload = client.post(f"/projects/{project_id}/video", files={"video": ("sample.mp4", handle, "video/mp4")})
        assert upload.status_code == 200, upload.text
        assert upload.json()["duration"] > 3.5
        assert upload.json()["width"] == 320
        assert upload.json()["original_filename"] == "sample.mp4"
        assert (projects / project_id / "source.mp4").is_file()
        video = client.get(f"/projects/{project_id}/video", headers={"Range": "bytes=10-19"})
        assert video.status_code == 206
        assert len(video.content) == 10
        assert video.headers["content-range"].startswith("bytes 10-19/")
        suffix = client.get(f"/projects/{project_id}/video", headers={"Range": "bytes=-5"})
        assert suffix.status_code == 206 and len(suffix.content) == 5
        invalid = client.get(f"/projects/{project_id}/video", headers={"Range": "bytes=999999999-"})
        assert invalid.status_code == 416

        audio = pipeline.extract_audio(project_id)
        assert audio.is_file()
        before = audio.stat().st_mtime_ns
        assert pipeline.extract_audio(project_id).stat().st_mtime_ns == before
        metadata = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(audio)], capture_output=True, text=True, check=True)
        stream = json.loads(metadata.stdout)["streams"][0]
        assert stream["channels"] == 1 and stream["sample_rate"] == "16000"

        folder = projects / project_id
        storage.atomic_json(folder / "transcript.json", {"segments": [{"start": 0, "end": 2, "text": "A good idea"}, {"start": 2, "end": 4, "text": "with a conclusion"}]})
        created_moment = client.post(f"/projects/{project_id}/moments", json={"start": 0.3, "end": 2.5, "title": "Sample clip"})
        assert created_moment.status_code == 200, created_moment.text
        moment = created_moment.json()
        assert moment["selected"] and "A good idea" in moment["excerpt"]
        updated = client.patch(f"/projects/{project_id}/moments/{moment['id']}", json={"end": 3.0, "selected": False})
        assert updated.status_code == 200 and updated.json()["end"] == 3.0
        assert client.get(f"/projects/{project_id}/moments").json()["moments"][0]["selected"] is False
        assert client.post(f"/projects/{project_id}/clips", json={"moment_ids": [moment["id"]]}).status_code == 422
        client.patch(f"/projects/{project_id}/moments/{moment['id']}", json={"selected": True})
        exported = client.post(f"/projects/{project_id}/clips", json={"moment_ids": [moment["id"]]})
        assert exported.status_code == 200, exported.text
        clip = exported.json()["clips"][0]
        assert client.get(clip["url"]).status_code == 200
        assert (folder / "clips" / clip["filename"]).stat().st_size > 0
        assert source.stat().st_size == (folder / "source.mp4").stat().st_size


def test_chunks_and_deduplication():
    segments = [{"start": i * 60, "end": (i + 1) * 60, "text": f"segment {i}"} for i in range(40)]
    chunks = pipeline.transcript_chunks(segments)
    assert len(chunks) >= 3
    assert chunks[0][-1]["start"] >= chunks[1][0]["start"]
    first = {"start": 750, "end": 800, "score": 90}
    duplicate = {"start": 755, "end": 805, "score": 70}
    different = {"start": 802, "end": 850, "score": 80}
    assert pipeline.deduplicate([first, duplicate, different]) == [first, different]


def test_analysis_context_refinement_dedup_and_cache(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    folder = projects / "abcdef123456"
    folder.mkdir(parents=True)
    monkeypatch.setattr(storage, "PROJECTS", projects)
    monkeypatch.setattr(main, "PROJECTS", projects)
    storage.atomic_json(folder / "project.json", {"id": folder.name, "duration": 1900, "max_clip_duration": 60, "state": "CREATED", "stages": {}})
    (folder / "source.mp4").write_bytes(b"placeholder")
    segments = [{"start": i * 30, "end": (i + 1) * 30, "text": f"Idea {i}."} for i in range(64)]
    storage.atomic_json(folder / "transcript.json", {"segments": segments})
    manual = {"id": "manual-clip", "start": 100, "end": 130, "title": "Manual idea", "source": "manual", "selected": True}
    storage.atomic_json(folder / "moments.json", {"moments": [manual]})
    calls = []

    def fake_llm(system, payload, *_args):
        calls.append(system)
        if "this portion" in system:
            return {"summary": "A series of ideas", "topics": ["ideas"], "participants_context": "One speaker"}
        if "Combine summaries" in system:
            assert len(payload["chunk_summaries"]) == 2
            return {"session_type": "lecture", "session_summary": "A discussion of ideas", "main_topics": ["ideas"], "participants_context": "One speaker"}
        if "Find ALL" in system:
            start = payload["chunk_start"]
            return {"moments": [{"start": start + 10, "end": start + 90, "title": "An idea", "category": "educational", "score": 88, "reason": "Standalone"}]}
        if "Refine" in system:
            return {"start": payload["candidate"]["start"] + 1, "end": payload["candidate"]["end"] - 1}
        raise AssertionError(system)

    monkeypatch.setattr(pipeline, "ask_llm", fake_llm)
    monkeypatch.setattr(pipeline, "ask_llm_with_model", lambda *args: (fake_llm(*args), "test/model:free"))
    result = pipeline.analyze(folder.name)
    assert len(result) == 3
    assert result[-1] == manual
    assert all(moment["end"] - moment["start"] <= 60 for moment in result)
    assert storage.read_json(folder / "global-context.json")["session_type"] == "lecture"
    assert len(storage.read_json(folder / "context-chunks.json")["summaries"]) == 2
    chunk_records = list(storage.read_json(folder / "analysis-chunks.json")["chunks"].values())
    assert [item["status"] for item in chunk_records] == ["completed", "completed"]
    assert all(item["summary_model"] == "test/model:free" and item["model"] == "test/model:free" for item in chunk_records)
    assert storage.read_json(folder / "project.json")["state"] == "READY"
    count = len(calls)
    assert pipeline.analyze(folder.name) == result
    assert len(calls) == count


def test_openrouter_retries_and_reasoning_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODELS", "test/first:free,test/second:free")
    requests = []

    class FakeResponse:
        def __init__(self, status, body):
            self.status_code = status
            self.body = body
            self.headers = {}
            self.is_error = status >= 400
            self.text = json.dumps(body)

        def json(self):
            return self.body

    responses = iter([
        FakeResponse(400, {"error": {"message": "Reasoning is mandatory and cannot be disabled"}}),
        FakeResponse(429, {"error": {"message": "Temporarily rate limited"}}),
        FakeResponse(200, {"choices": [{"message": {"content": '{"moments":[]}'}}]}),
    ])

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, _url, *, json, **_kwargs):
            requests.append(dict(json))
            return next(responses)

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)
    result = pipeline.ask_llm("Find moments", {"segments": []}, pipeline.MOMENTS_SCHEMA, "moments")
    assert result == {"moments": []}
    assert "reasoning" in requests[0]
    assert "reasoning" not in requests[1]
    assert requests[2]["response_format"]["type"] == "json_schema"
    assert [item["model"] for item in requests] == ["test/first:free", "test/first:free", "test/second:free"]
    assert all(item["messages"] == requests[0]["messages"] and item["response_format"] == requests[0]["response_format"] for item in requests)


def test_forced_transcription_invalidates_ai_outputs_but_preserves_manual_clips(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    folder = projects / "abcdef123456"
    folder.mkdir(parents=True)
    monkeypatch.setattr(storage, "PROJECTS", projects)
    storage.atomic_json(folder / "project.json", {"id": folder.name, "state": "READY", "stages": {"transcription": "completed", "global_context": "completed", "analysis": "completed"}})
    storage.atomic_json(folder / "transcript.json", {"segments": [{"start": 0, "end": 1, "text": "Old"}]})
    storage.atomic_json(folder / "context-chunks.json", {"summaries": []})
    storage.atomic_json(folder / "global-context.json", {"session_summary": "Old"})
    storage.atomic_json(folder / "analysis-refinements.json", {"refined": {"0": {"title": "Old"}}})
    manual = {"id": "manual", "source": "manual"}
    storage.atomic_json(folder / "moments.json", {"moments": [{"id": "ai", "source": "ai"}, manual]})
    monkeypatch.setattr(pipeline, "extract_audio", lambda _project_id: folder / "audio.wav")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"segments": [{"start": 0, "end": 1, "text": "New"}]}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)
    assert pipeline.transcribe(folder.name, force=True)["segments"][0]["text"] == "New"
    assert not (folder / "global-context.json").exists()
    assert not (folder / "context-chunks.json").exists()
    assert not (folder / "analysis-refinements.json").exists()
    assert storage.read_json(folder / "moments.json")["moments"] == [manual]
    assert storage.read_json(folder / "project.json")["stages"]["analysis"] == "pending"
    assert storage.read_json(folder / "project.json")["state"] == "CREATED"


def test_analysis_starts_from_partial_transcript_before_whisper_finishes(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    folder = projects / "abcdef123456"
    folder.mkdir(parents=True)
    monkeypatch.setattr(storage, "PROJECTS", projects)
    storage.atomic_json(folder / "project.json", {"id": folder.name, "duration": 1500, "max_clip_duration": 60, "state": "CREATED", "stages": {}})
    started_analysis = threading.Event()
    finished_transcription = threading.Event()
    partial_segment = {"start": 0, "end": 1200, "text": "An important explanation."}
    final_segment = {"start": 1200, "end": 1500, "text": "A second explanation."}

    def fake_transcribe(_project_id):
        (folder / "transcript.partial.jsonl").write_text(json.dumps(partial_segment) + "\n")
        storage.atomic_json(folder / "transcription-progress.json", {"processed_audio_seconds": 1320, "total_audio_seconds": 1500, "progress_percentage": 88, "status": "running"})
        assert started_analysis.wait(5), "Analysis did not start before transcription finished"
        finished_transcription.set()
        result = {"segments": [partial_segment, final_segment]}
        storage.atomic_json(folder / "transcript.json", result)
        return result

    def fake_llm(system, payload, *_args):
        if "Summarize this portion" in system:
            return {"summary": "Two explanations", "topics": ["ideas"], "participants_context": "Teacher"}
        if "Combine summaries" in system:
            return {"session_type": "class", "session_summary": "Two explanations", "main_topics": ["ideas"], "participants_context": "Teacher"}
        if "Find ALL" in system:
            if payload["chunk_start"] == 0:
                assert not finished_transcription.is_set()
                started_analysis.set()
            return {"moments": [{"start": payload["chunk_start"] + 10, "end": payload["chunk_start"] + 70, "title": "Idea", "category": "educational", "score": 85, "reason": "Useful"}]}
        if "Refine" in system:
            return {"start": payload["candidate"]["start"], "end": payload["candidate"]["end"]}
        raise AssertionError(system)

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "ask_llm", fake_llm)
    monkeypatch.setattr(pipeline, "ask_llm_with_model", lambda *args: (fake_llm(*args), "test/model:free"))
    result = pipeline.analyze(folder.name)
    assert finished_transcription.is_set()
    assert len(result) == 2
    assert [item["status"] for item in storage.read_json(folder / "analysis-chunks.json")["chunks"].values()] == ["completed", "completed"]


def test_completed_chunk_is_not_repeated_after_retry(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    folder = projects / "abcdef123456"
    folder.mkdir(parents=True)
    monkeypatch.setattr(storage, "PROJECTS", projects)
    storage.atomic_json(folder / "project.json", {"id": folder.name, "duration": 1500, "max_clip_duration": 60, "state": "FAILED", "stages": {}})
    segments = [{"start": 0, "end": 1200, "text": "First chunk"}, {"start": 1200, "end": 1500, "text": "Second chunk"}]
    storage.atomic_json(folder / "transcript.json", {"segments": segments})
    storage.atomic_json(folder / "analysis-chunks.json", {"chunks": {"0": {"start": 0, "end": 1200, "status": "completed", "summary": {"summary": "First", "topics": [], "participants_context": ""}, "candidates": []}}})
    detections = []

    def fake_llm(system, payload, *_args):
        if "Summarize this portion" in system:
            return {"summary": "Second", "topics": [], "participants_context": ""}
        if "Combine summaries" in system:
            return {"session_type": "class", "session_summary": "Both chunks", "main_topics": [], "participants_context": ""}
        if "Find ALL" in system:
            detections.append(payload["chunk_start"])
            return {"moments": []}
        raise AssertionError(system)

    monkeypatch.setattr(pipeline, "ask_llm", fake_llm)
    monkeypatch.setattr(pipeline, "ask_llm_with_model", lambda *args: (fake_llm(*args), "test/model:free"))
    pipeline.analyze(folder.name)
    assert detections == [1080]


def test_429_falls_through_free_models_and_preserves_retry_after(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODELS", "test/first:free,test/second:free")
    delays = []
    monkeypatch.setattr(pipeline.time, "sleep", delays.append)

    class FakeResponse:
        status_code = 429
        headers = {"Retry-After": "17"}
        is_error = True
        text = "rate limited"

        def json(self):
            return {"error": {"message": "rate limited"}}

    class FakeClient:
        def __init__(self, **_kwargs):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, *_args, **kwargs):
            self.calls.append(kwargs["json"]["model"])
            sent.extend(self.calls[-1:])
            return FakeResponse()

    sent = []
    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)
    with pytest.raises(pipeline.OpenRouterModelsExhausted) as caught:
        pipeline.ask_llm("Find moments", {"segments": []})
    assert sent == ["test/first:free", "test/second:free"]
    assert caught.value.retry_after == 17
    assert delays == []


def test_429_pauses_analysis_while_transcription_finishes(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    folder = projects / "abcdef123456"
    folder.mkdir(parents=True)
    monkeypatch.setattr(storage, "PROJECTS", projects)
    storage.atomic_json(folder / "project.json", {"id": folder.name, "duration": 1500, "max_clip_duration": 60, "state": "CREATED", "stages": {}})
    segment = {"start": 0, "end": 1200, "text": "Partial transcript"}
    finished = threading.Event()

    def fake_transcribe(_project_id):
        (folder / "transcript.partial.jsonl").write_text(json.dumps(segment) + "\n")
        storage.atomic_json(folder / "transcription-progress.json", {"processed_audio_seconds": 1320, "total_audio_seconds": 1500})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            path = folder / "analysis-chunks.json"
            if path.exists() and storage.read_json(path)["chunks"].get("0", {}).get("status") == "pending":
                break
            time.sleep(0.01)
        else:
            raise AssertionError("Analysis was not paused")
        finished.set()
        result = {"segments": [segment, {"start": 1200, "end": 1500, "text": "Final transcript"}]}
        storage.atomic_json(folder / "transcript.json", result)
        return result

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "ANALYSIS_RETRY_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(pipeline, "ask_llm_with_model", lambda *_args: (_ for _ in ()).throw(pipeline.OpenRouterModelsExhausted(["test/model:free: OpenRouter 429"])))
    with pytest.raises(pipeline.OpenRouterModelsExhausted):
        pipeline.analyze(folder.name)
    assert finished.is_set()
    assert (folder / "transcript.partial.jsonl").exists()
    assert (folder / "transcript.json").exists()
    assert storage.read_json(folder / "analysis-chunks.json")["chunks"]["0"]["status"] == "pending"
    assert storage.read_json(folder / "project.json")["state"] == "ANALYSIS_PAUSED"


def test_sse_emits_progress_and_partial_segments(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    folder = projects / "abcdef123456"
    folder.mkdir(parents=True)
    monkeypatch.setattr(storage, "PROJECTS", projects)
    monkeypatch.setattr(main, "PROJECTS", projects)
    storage.atomic_json(folder / "project.json", {"id": folder.name, "state": "TRANSCRIBING", "stages": {"transcription": "running"}})
    storage.atomic_json(folder / "transcription-progress.json", {"processed_audio_seconds": 30, "total_audio_seconds": 300, "progress_percentage": 10, "realtime_speed": 2, "estimated_remaining_seconds": 135})
    (folder / "transcript.partial.jsonl").write_text('{"start":0,"end":5,"text":"Hello"}\n')
    with TestClient(main.app) as client:
        assert client.get(f"/projects/{folder.name}").json()["transcription_progress"]["progress_percentage"] == 10

    async def read_two():
        request = Request({"type": "http", "method": "GET", "path": f"/projects/{folder.name}/events", "headers": []})
        response = await main.events(folder.name, request)
        stream = response.body_iterator
        first = await anext(stream)
        second = await anext(stream)
        await stream.aclose()
        return first, second

    first, second = asyncio.run(read_two())
    assert '"progress_percentage": 10' in first
    assert "event: transcript_segment" in second
    assert '"text":"Hello"' in second

    (folder / "transcript.partial.jsonl").write_text('{"start":0,"end":5,"text":"Hello"}\n{"start":5,"end":10,"text":"World"}\n')

    async def resume_after_first():
        request = Request({"type": "http", "method": "GET", "path": f"/projects/{folder.name}/events", "headers": [(b"last-event-id", b"segment-1")]})
        response = await main.events(folder.name, request)
        stream = response.body_iterator
        await anext(stream)
        next_segment = await anext(stream)
        await stream.aclose()
        return next_segment

    resumed = asyncio.run(resume_after_first())
    assert "id: segment-2" in resumed
    assert '"text":"World"' in resumed


def test_process_exposes_paused_state_on_rate_limit(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    folder = projects / "123456abcdef"
    folder.mkdir(parents=True)
    monkeypatch.setattr(storage, "PROJECTS", projects)
    monkeypatch.setattr(main, "PROJECTS", projects)
    storage.atomic_json(folder / "project.json", {"id": folder.name, "state": "CREATED", "stages": {"analysis": "pending"}})
    (folder / "source.mp4").write_bytes(b"source")
    monkeypatch.setattr(pipeline, "analyze", lambda _project_id: (_ for _ in ()).throw(pipeline.OpenRouterRateLimit("OpenRouter 429")))
    with TestClient(main.app) as client:
        assert client.post(f"/projects/{folder.name}/process").status_code == 202
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = client.get(f"/projects/{folder.name}").json()
            if state["state"] == "ANALYSIS_PAUSED":
                break
            time.sleep(0.01)
    assert state["state"] == "ANALYSIS_PAUSED"
    assert state["stages"]["analysis"] == "paused"


def test_openrouter_requests_are_sequential(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODELS", "test/model:free")
    active = 0
    maximum = 0
    guard = threading.Lock()

    class FakeResponse:
        status_code = 200
        is_error = False

        def json(self):
            return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, *_args, **_kwargs):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return FakeResponse()

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)
    workers = [threading.Thread(target=pipeline.ask_llm, args=("Respond", {})) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert maximum == 1
