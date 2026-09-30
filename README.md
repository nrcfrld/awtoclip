# Awtoclip

A local web app for turning long recordings into reviewable short video clips. Video and audio stay on your machine. The transcript is sent to OpenRouter for context, moment detection, and boundary refinement.

## Requirements

- macOS on Apple Silicon for the native MLX Whisper service
- Docker Desktop with Docker Compose
- An OpenRouter API key and free models that support JSON responses
- Free disk space for the original video, a 16 kHz WAV, and exported clips

The first Whisper run downloads the configured model from Hugging Face. After that, transcription runs locally on the Mac. The default is [`mlx-community/whisper-small-mlx`](https://huggingface.co/mlx-community/whisper-small-mlx); it can be changed with `WHISPER_MODEL`.

## Start

1. Copy `.env.example` to `.env`. Set `OPENROUTER_API_KEY`, `OPENROUTER_MODELS` (ordered, comma-separated free model IDs), and `HOST_DATA_ROOT` to the absolute path of this repository's `data` directory. Leave `DATA_ROOT=/data` and `WHISPER_SERVICE_URL=http://host.docker.internal:9001` for Docker. Every model must end in `:free`; `openrouter/free` is also allowed. Paid model IDs are rejected.
2. Start the native Whisper service in a separate terminal on the Mac:

   ```bash
   cd apps/whisper-service
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   HOST_DATA_ROOT="/absolute/path/to/awtoclip/data" WHISPER_MODEL="mlx-community/whisper-small-mlx" .venv/bin/uvicorn main:app --host 0.0.0.0 --port 9001
   ```

   Use the same absolute `HOST_DATA_ROOT` as in `.env`. The service listens on the host so the API container can reach it through `host.docker.internal`. Use this stack on a trusted local network.

3. From the repository root, start the web stack:

   ```bash
   docker compose up --build
   ```

4. Open [http://localhost:5173](http://localhost:5173). The API health endpoint is [http://localhost:8000/health](http://localhost:8000/health); the Whisper health endpoint is [http://localhost:9001/health](http://localhost:9001/health).

## Workflow

Select an MP4, MOV, or MKV and choose the maximum clip duration (60 seconds by default). The browser uploads the file in a streaming request. The project copy becomes the source for playback and exports. Processing extracts 16 kHz mono audio. Native Whisper processes it in 30 second windows and appends finalized segments to the live transcript. The UI displays measured audio progress, realtime speed, and an ETA once at least 30 seconds of processing or 10% of the audio has completed.

AI analysis uses 20 minute transcript chunks with a 2 minute overlap and a 2 minute lookahead. A full chunk can start as soon as Whisper has processed through its end plus the lookahead; the final shorter chunk runs after transcription completes. OpenRouter requests run one at a time. After transcription, the app combines the chunk summaries into global context, refines clip boundaries, and deduplicates results.

Preview moments from the source video without rendering temporary clips. Select, edit, or reject candidates, search and click the synchronized transcript, or mark a manual start and end. Export one moment or all selected moments as MP4 files in `data/projects/{project-id}/clips/`. Exported files also appear as download links in the app.

Projects and intermediate JSON outputs persist under `data/projects/`. Reopening a project loads those outputs. A failed or interrupted processing stage can be retried with **Retry processing**; completed audio, transcription, context, and analysis chunks are reused. Each OpenRouter request tries the configured free models in order on rate limits, timeouts, or provider outages, with the same prompt and JSON schema. Completed chunks record the models used for summarization and moment detection. If every model fails, analysis pauses with its current chunk pending while local Whisper continues. The app waits at least 10 seconds (or longer when `Retry-After` requires it) and retries the pending queue once. If that also fails, use **Retry processing** after the upstream limit clears.

## Project files

```text
data/projects/{project-id}/
  source.{mp4,mov,mkv}
  project.json
  audio.wav
  transcript.partial.jsonl
  transcription-progress.json
  transcript.json
  analysis-chunks.json
  analysis-refinements.json
  context-chunks.json
  global-context.json
  moments.json
  clips/*.mp4
```

## Development and tests

The API is in `apps/api`, the React/Vite UI in `apps/web`, the native Whisper service in `apps/whisper-service`, and Compose config in `infra/docker-compose.yml`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r apps/api/requirements.txt pytest
.venv/bin/python -m pytest -q apps/api/tests
apps/whisper-service/.venv/bin/python -m unittest discover -s apps/whisper-service -p 'test_*.py'
cd apps/web && bun install && bun run build
```

The tests use FFmpeg and ffprobe installed on the host. To run the API outside Docker, set `DATA_ROOT` to this repository's `data` directory and `WHISPER_SERVICE_URL=http://localhost:9001`, then run `uvicorn app.main:app --app-dir apps/api --port 8000`. For the Vite development server, run `bun run dev` in `apps/web`; its `/api` proxy targets port 8000.

## API

The interactive API reference is at [http://localhost:8000/docs](http://localhost:8000/docs). Main routes include project creation/listing, video upload and HTTP Range playback, stage triggers, transcript and moment retrieval, candidate editing and creation, MP4 export, and SSE progress plus partial transcript segments at `GET /projects/{id}/events`. The native Whisper service also exposes its live metrics at `GET http://localhost:9001/projects/{id}/progress`.

No authentication is included; this is a single-user local MVP. Do not expose ports 5173, 8000, or 9001 to an untrusted network.
