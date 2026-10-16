from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field, ConfigDict
from scipy.io import wavfile
from scipy.signal import resample_poly
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel, pipeline

# Load project-local configuration for the documented `uvicorn app:app` workflow.
load_dotenv(Path(__file__).with_name(".env"))

# ---------------- Configuration ----------------
APP_NAME = os.getenv("APP_NAME", "Dinka AI API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
# Treat an empty/whitespace token as unset. Passing token="" makes
# huggingface_hub send the invalid header "Authorization: Bearer ".
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
MT_MODEL = os.getenv("MT_MODEL", "Alaak/dinka-nllb-600m")
MT_FALLBACK = os.getenv("MT_FALLBACK", "facebook/nllb-200-distilled-600M")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
TTS_MODEL = os.getenv("TTS_MODEL", "facebook/mms-tts-dik")
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "").split(",") if x.strip()]
EN_LANG = os.getenv("EN_LANG", "eng_Latn")
DIK_LANG = os.getenv("DIK_LANG", "dik_Latn")
MAX_SESSION_TURNS = int(os.getenv("MAX_SESSION_TURNS", "20"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "2000"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
MAX_AUDIO_CHARS = int(os.getenv("MAX_AUDIO_CHARS", "400"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "256"))
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-small")
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))
CORRECTION_QUEUE_PATH = Path(os.getenv("CORRECTION_QUEUE_PATH", "data/correction_submissions.jsonl"))
DEVICE = "cuda" if torch.cuda.is_available() and os.getenv("FORCE_CPU", "0") != "1" else "cpu"

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are Dinka AI, a warm and respectful assistant for South Sudanese people. "
    "Reply in short, simple English sentences, two to four sentences at most, because "
    "your reply may be translated into Dinka. Avoid idioms, slang, and long lists. "
    "For medical, legal, or financial topics, provide general information and encourage "
    "professional advice where appropriate. Never claim certainty when uncertain."
)
GREETING = "Loidi"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("dinka_ai")

# ---------------- Schemas ----------------
class ChatIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=1000)
    lang: Literal["en", "dik"] = "en"
    session_id: str | None = Field(default=None, max_length=128)
    model: str | None = Field(default=None, max_length=200)

class TranslateIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=2000)
    direction: Literal["en2dik", "dik2en"]

class SpeakIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=MAX_AUDIO_CHARS)

class CorrectionIn(BaseModel):
    english: str = Field(min_length=1, max_length=2000)
    dinka_bor: str = Field(min_length=1, max_length=2000)
    model_output: str = Field(default="", max_length=2000)
    kind: str = Field(default="reply", max_length=40)

class CorrectionSubmissionIn(BaseModel):
    english: str = Field(min_length=1, max_length=2000)
    dinka_bor: str = Field(min_length=1, max_length=2000)
    model_output: str = Field(default="", max_length=2000)
    kind: str = Field(default="reply", max_length=40)

# ---------------- Runtime state ----------------
mt_tokenizer = None
mt_model = None
mt_loaded_name = None
tts_tokenizer = None
tts_model = None
llm_client = None
asr_pipeline = None
asr_lock = threading.RLock()
correction_rate_lock = threading.RLock()
correction_rate: dict[str, list[float]] = {}

mt_lock = threading.RLock()
tts_lock = threading.RLock()
llm_lock = threading.RLock()
sessions_lock = threading.RLock()
sessions: OrderedDict[str, dict] = OrderedDict()
corrections: dict[str, str] = {}

BOR_UI = {
    "hi": "Loidi",
    "sub": "Yeŋö lëu bï ya kony rötdu tëne?",
    "empty_sub": "Yeŋö lëu bï ya kony rötdu tëne?",
    "ph": "Cöör wël",
    "send": "Cöör",
    "translate": "Gɛ̈ɛ̈r",
    "speak": "Cöör",
}

def _load_translation_model():
    global mt_tokenizer, mt_model, mt_loaded_name
    last_error = None
    candidates = list(dict.fromkeys([MT_MODEL, MT_FALLBACK]))
    for name in candidates:
        try:
            log.info("Loading translation model %s on %s", name, DEVICE)
            tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
            model = AutoModelForSeq2SeqLM.from_pretrained(name, token=HF_TOKEN)
            model.to(DEVICE).eval()
            if not hasattr(tok, "lang_code_to_id") and tok.convert_tokens_to_ids(DIK_LANG) == tok.unk_token_id:
                raise RuntimeError(
                    f"Tokenizer {name} does not recognize target language tag {DIK_LANG!r}. "
                    "Use a Dinka-compatible fine-tuned NLLB tokenizer/model or set DIK_LANG correctly."
                )
            mt_tokenizer, mt_model, mt_loaded_name = tok, model, name
            log.info("Translation model loaded: %s", name)
            return
        except Exception as exc:
            last_error = exc
            log.exception("Could not load translation model %s", name)
    raise RuntimeError(f"Could not load any translation model: {last_error}")

def _load_tts():
    global tts_tokenizer, tts_model
    log.info("Loading TTS model %s on %s", TTS_MODEL, DEVICE)
    tok = AutoTokenizer.from_pretrained(TTS_MODEL)
    model = VitsModel.from_pretrained(TTS_MODEL)
    model.to(DEVICE).eval()
    tts_tokenizer, tts_model = tok, model
    log.info("TTS model loaded")

def _prune_sessions(now: float | None = None):
    now = now or time.time()
    expired = [sid for sid, item in sessions.items() if now - item["updated"] > SESSION_TTL_SECONDS]
    for sid in expired:
        sessions.pop(sid, None)
    while len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_translation_model()
    _load_tts()
    global llm_client
    llm_client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN, timeout=60)
    yield
    log.info("Application shutdown")

app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Admin-Key"],
    )

def require_admin(x_admin_key: str | None = Header(default=None)):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin endpoints are disabled: ADMIN_API_KEY is not configured")
    if not x_admin_key or not __import__("hmac").compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid admin key")

def _lang_id(tokenizer, language: str) -> int:
    # NLLB tokenizers expose lang_code_to_id in common Transformers versions.
    mapping = getattr(tokenizer, "lang_code_to_id", None)
    if mapping and language in mapping:
        return mapping[language]
    token_id = tokenizer.convert_tokens_to_ids(language)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise RuntimeError(
            f"Language code {language!r} is not supported by the loaded tokenizer. "
            "Check the fine-tuned model and DIK_LANG environment variable."
        )
    return token_id

def translate_text(text: str, src: str, tgt: str) -> str:
    if not text.strip():
        return ""
    with mt_lock, torch.inference_mode():
        mt_tokenizer.src_lang = src
        inputs = mt_tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        generated = mt_model.generate(
            **inputs,
            forced_bos_token_id=_lang_id(mt_tokenizer, tgt),
            max_new_tokens=256,
            max_length=None,
            num_beams=4,
        )
        return mt_tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

def to_dinka(text: str) -> str:
    key = text.strip().lower()
    if key in corrections:
        return corrections[key]
    if key in BOR_UI:
        return BOR_UI[key]
    return translate_text(text, EN_LANG, DIK_LANG)

def synthesize(text: str) -> bytes:
    text = text.strip()[:MAX_AUDIO_CHARS]
    if not text:
        raise ValueError("Nothing to speak")
    with tts_lock, torch.inference_mode():
        inputs = tts_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        waveform = tts_model(**inputs).waveform.squeeze().detach().cpu().numpy()
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    output = io.BytesIO()
    wavfile.write(output, int(tts_model.config.sampling_rate), pcm)
    return output.getvalue()

def _get_history(session_id: str) -> list[dict]:
    with sessions_lock:
        _prune_sessions()
        item = sessions.get(session_id)
        if not item:
            return []
        item["updated"] = time.time()
        sessions.move_to_end(session_id)
        return list(item["history"])

def _save_history(session_id: str, history: list[dict]):
    with sessions_lock:
        sessions[session_id] = {"history": history[-MAX_SESSION_TURNS:], "updated": time.time()}
        sessions.move_to_end(session_id)
        _prune_sessions()

def _ask_llm(history: list[dict], requested_model: str | None = None) -> tuple[str, str]:
    model_id = requested_model or LLM_MODEL
    # A user-supplied model ID is not allowed to select arbitrary remote models.
    if model_id != LLM_MODEL:
        raise HTTPException(status_code=400, detail="Requested model is not enabled by this deployment")
    if not HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Hugging Face access is not configured. Set HF_TOKEN in the project .env file or server environment.",
        )
    with llm_lock:
        try:
            result = llm_client.chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                max_tokens=300,
                temperature=0.4,
            )
        except Exception as exc:
            error_text = str(exc).casefold()
            if "no remaining credits" in error_text or (
                "inference providers" in error_text and "credits" in error_text
            ):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Hugging Face Inference Providers has no remaining credits for this account. "
                        "Add provider credits, wait for included credits to renew, or configure a different LLM service."
                    ),
                ) from exc
            raise
    content = result.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("Language model returned an empty response")
    return content.strip(), model_id

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(Path(__file__).parent / "frontend" / "index.html", media_type="text/html")

@app.get("/api", include_in_schema=False)
def api_root():
    return {"service": APP_NAME, "version": APP_VERSION, "docs": "/docs"}

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "translation_model": mt_loaded_name,
        "voice_model": TTS_MODEL,
        "llm": LLM_MODEL,
    }

@app.get("/api/greeting")
def greeting():
    return {"dinka": GREETING, "english": "Hello, I am Dinka AI. Talk to me in Dinka or English."}

@app.get("/api/ui_strings")
def ui_strings():
    merged = dict(BOR_UI)
    merged.update(corrections)
    return {"lang": "din", "strings": merged, "count": len(merged)}

@app.post("/api/translate")
def translate_endpoint(body: TranslateIn):
    try:
        if body.direction == "en2dik":
            return {"translation": to_dinka(body.text), "model": mt_loaded_name}
        return {"translation": translate_text(body.text, DIK_LANG, EN_LANG), "model": mt_loaded_name}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Translation failed")
        raise HTTPException(status_code=502, detail="Translation service failed") from exc

@app.post("/api/chat")
def chat(body: ChatIn):
    session_id = body.session_id or uuid.uuid4().hex
    try:
        user_english = translate_text(body.message, DIK_LANG, EN_LANG) if body.lang == "dik" else body.message
        history = _get_history(session_id)
        history.append({"role": "user", "content": user_english})
        reply_english, used_model = _ask_llm(history, body.model)
        history.append({"role": "assistant", "content": reply_english})
        _save_history(session_id, history)
        reply_dinka = to_dinka(reply_english)
        return {
            "session_id": session_id,
            "user_english": user_english,
            "reply_english": reply_english,
            "reply_dinka": reply_dinka,
            "model_used": used_model,
            "translation_model": mt_loaded_name,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Chat failed")
        raise HTTPException(status_code=502, detail="Chat service failed") from exc

@app.post("/api/speak")
def speak_endpoint(body: SpeakIn):
    try:
        return Response(content=synthesize(body.text), media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Speech synthesis failed")
        raise HTTPException(status_code=502, detail="Speech synthesis failed") from exc

def _get_asr_pipeline():
    global asr_pipeline
    with asr_lock:
        if asr_pipeline is None:
            log.info("Loading ASR model %s on %s", ASR_MODEL, DEVICE)
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=ASR_MODEL,
                device=0 if DEVICE == "cuda" else -1,
                token=HF_TOKEN,
            )
        return asr_pipeline

@app.post("/api/listen")
async def listen_endpoint(request: Request):
    """Accept a browser-recorded WAV body and return transcribed text."""
    content_type = request.headers.get("content-type", "").lower()
    if "audio/" not in content_type and "application/octet-stream" not in content_type:
        raise HTTPException(status_code=415, detail="Send audio as audio/wav")
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Audio body is empty")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file is too large")
    try:
        sample_rate, audio = wavfile.read(io.BytesIO(raw))
        if audio.ndim == 2:
            audio = audio.astype(np.float32).mean(axis=1)
        if np.issubdtype(audio.dtype, np.integer):
            max_val = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)
        if sample_rate != 16000:
            import math
            divisor = math.gcd(int(sample_rate), 16000)
            audio = resample_poly(audio, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
            sample_rate = 16000
        if audio.size < 1600:
            raise HTTPException(status_code=422, detail="Recording is too short")
        result = _get_asr_pipeline()({"raw": audio, "sampling_rate": sample_rate})
        text = (result.get("text") or "").strip()
        return {"text": text}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Audio transcription failed")
        raise HTTPException(status_code=502, detail="Audio transcription failed. Check ASR_MODEL and model support.") from exc

@app.post("/api/correction-submissions", status_code=202)
def submit_correction(body: CorrectionSubmissionIn, request: Request):
    """Public correction intake; suggestions are queued, never applied automatically."""
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    with correction_rate_lock:
        recent = [t for t in correction_rate.get(ip, []) if now - t < 3600]
        if len(recent) >= 5:
            raise HTTPException(status_code=429, detail="Correction limit reached. Try again later.")
        recent.append(now)
        correction_rate[ip] = recent
    record = {
        "id": uuid.uuid4().hex,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "english": body.english.strip(),
        "model_output": body.model_output.strip(),
        "dinka_bor": body.dinka_bor.strip(),
        "kind": body.kind.strip()[:40],
    }
    try:
        CORRECTION_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CORRECTION_QUEUE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.exception("Could not save correction submission")
        raise HTTPException(status_code=503, detail="Correction queue is temporarily unavailable") from exc
    return {"submitted": True, "message": "Thank you. Your suggestion was sent for review."}

@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):
    with sessions_lock:
        existed = sessions.pop(session_id, None) is not None
    return {"cleared": True, "session_existed": existed}

@app.post("/api/correct", dependencies=[Depends(require_admin)])
def correct_endpoint(body: CorrectionIn):
    key = body.english.strip().lower()
    corrections[key] = body.dinka_bor.strip()
    return {"saved": True, "english_key": key, "dinka_bor": body.dinka_bor.strip(),
            "corrections_total": len(corrections)}

@app.get("/api/models")
def models():
    return {"default": LLM_MODEL, "models": [{"id": LLM_MODEL, "label": LLM_MODEL}]}
