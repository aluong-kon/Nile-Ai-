# Dinka AI — integrated web app + FastAPI backend

This project combines the uploaded Dinka AI chat interface with the FastAPI backend. The backend serves the frontend at `/` and provides chat, model selection, Dinka translation, speech synthesis, speech-to-text, session clearing, and secure correction intake.

**This is a production-oriented starter, not a certified production deployment.** Model availability, Dinka language support, speech recognition quality, licensing, security, and actual performance must be tested before public launch.

## Features and routes

- `GET /` — serves `frontend/index.html`
- `GET /api/health` — loaded model names and device
- `GET /api/greeting` — greeting
- `GET /api/ui_strings` — Dinka Bor UI glossary
- `GET /api/models` — configured chat model list
- `POST /api/translate` — English/Dinka translation
- `POST /api/chat` — LLM chat, with bounded in-memory session history
- `POST /api/speak` — returns WAV audio for text
- `POST /api/listen` — accepts raw WAV audio and returns `{ "text": "..." }`
- `DELETE /api/session/{session_id}` — clears a chat session
- `POST /api/correction-submissions` — public, rate-limited correction suggestion queue; submissions are not applied automatically
- `POST /api/correct` — admin-only direct glossary correction; requires `X-Admin-Key`
- `GET /docs` — interactive API documentation

## Run on Windows 11

Use Python 3.11 (recommended for broad ML package compatibility). In PowerShell, from the extracted project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, use this temporary setting for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open `.env` and set `HF_TOKEN` if your chosen model/provider requires it. Keep secrets in `.env` or your hosting provider's secret settings; never put them in HTML/JavaScript or commit `.env`.

Start the integrated app:

```powershell
uvicorn app:app --host 127.0.0.1 --port 7860 --workers 1
```

Open `http://127.0.0.1:7860`. API docs: `http://127.0.0.1:7860/docs`.

The first use may download several gigabytes of model weights. Translation and TTS models load at startup; the speech-to-text model loads on the first microphone transcription request. CPU-only systems may be slow.

## Voice input notes

The frontend records microphone audio in the browser, converts it to WAV, then posts it to `/api/listen`. Browser microphone access generally requires HTTPS or `localhost`. The server uses the configurable `ASR_MODEL` (default `openai/whisper-small`). Whisper's Dinka Bor transcription quality is not guaranteed; evaluate with native speakers and choose a model that actually supports the target speech. The endpoint enforces a maximum audio payload size.

## Secure correction workflow

The correction button does **not** send an admin secret to the browser. It posts suggestions to `/api/correction-submissions`, which writes a JSONL review queue under `data/correction_submissions.jsonl` and rate-limits submissions by client IP. Suggestions do not change model behavior automatically. Review submissions manually before incorporating them into a glossary or training corpus.

For administrators, set a long random `ADMIN_API_KEY` in `.env`. The protected `/api/correct` route requires that key in the `X-Admin-Key` header and is intended for trusted server-side/admin tools only. Never embed this key in frontend code. The public queue's rate limiter is in-memory and is not a substitute for reverse-proxy or platform-level rate limiting in a multi-instance deployment.

## Important model compatibility caveats

- Verify that the configured translation tokenizer really supports `DIK_LANG=dik_Latn`. Standard `facebook/nllb-200-distilled-600M` does not automatically gain Dinka support just because the environment variable is set. The app checks for the language token and will fail rather than pretending an unsupported language is supported.
- Verify Hugging Face inference access for `LLM_MODEL` and ensure its license and provider terms fit your deployment.
- Review the license and commercial-use terms for all model weights, especially `facebook/mms-tts-dik`, before using the project commercially.
- The chat backend translates Dinka input to English before sending it to the LLM, then translates the reply back. Translation quality is a core dependency.

## Tests and checks

```powershell
python -m py_compile app.py train.py tests/test_schemas.py
pytest -q
```

These checks validate Python syntax and schema tests; they do not verify live model downloads, speech recognition accuracy, real Dinka translation quality, or production readiness.

## Deployment

- Configure `ALLOWED_ORIGINS` only if serving the frontend from a separate origin. The integrated single-origin setup does not need browser CORS.
- Use HTTPS, request-size limits, rate limiting, monitoring, and a managed store for sessions if deploying publicly.
- The in-memory session store resets on restart and is not shared between instances.
- Keep Uvicorn to one worker on constrained hardware because every worker loads a separate copy of each model.
- Do not expose admin endpoints directly to untrusted clients.

## Fine-tuning

`train.py` is separate from the web service. Use a UTF-8 JSONL dataset containing reviewed English/Dinka pairs, one object per line:

```json
{"english":"Hello","dinka_bor":"Loidi"}
{"english":"Thank you","dinka_bor":"..."}
```

Only use speaker-reviewed data with appropriate consent and rights. Training intentionally stops if the tokenizer does not support `DIK_LANG`.
