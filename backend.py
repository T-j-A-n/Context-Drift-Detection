from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import numpy as np
import io
import asyncio
import os
import torch
import warnings
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.utils import logging as hf_logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# ── optional: coreference resolution via coreferee ──────────────────────────
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("coreferee")
    COREF_AVAILABLE = True
except Exception:
    nlp = None
    COREF_AVAILABLE = False

# ── optional: speaker diarization via pyannote ───────────────────────────────
# Requires: pip install pyannote.audio
# And a HuggingFace token with access to pyannote/speaker-diarization-3.1
# Set env var HF_TOKEN=<your token> before starting the server.
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    _hf_token = os.environ.get("HF_TOKEN")
    if _hf_token:
        diarizer = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=_hf_token,
        )
        DIARIZATION_AVAILABLE = True
    else:
        diarizer = None
        DIARIZATION_AVAILABLE = False
except Exception:
    diarizer = None
    DIARIZATION_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# ── ASR: Distil-Whisper ───────────────────────────────────────────────────────
MODEL_ID = "distil-whisper/distil-large-v3"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)
model.eval()

# ── Sentence embeddings for drift ────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


# ── helpers ───────────────────────────────────────────────────────────────────

def decode_audio(audio_bytes: bytes):
    """Convert raw WebM bytes → normalized 16kHz mono float32 numpy array."""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    # Normalize amplitude so quiet microphones still register clearly
    peak = np.abs(samples).max()
    if peak > 0.001:
        samples = samples / peak * 0.95
    return samples


def is_hallucination(text: str) -> bool:
    """Reject Whisper repetition artifacts."""
    if not text:
        return False
    if len(text) > 8:
        top_char_ratio = max(text.count(c) for c in set(text)) / len(text)
        if top_char_ratio > 0.4:
            return True
    words = text.split()
    if len(words) > 4 and len(set(words)) / len(words) < 0.25:
        return True
    return False


def transcribe(audio_bytes: bytes) -> str:
    """Step 1 — Distil-Whisper ASR."""
    if len(audio_bytes) < 500:
        return ""
    try:
        samples = decode_audio(audio_bytes)
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                inputs.input_features,
                language="en",
                task="transcribe",
            )
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return "" if is_hallucination(text) else text
    except CouldntDecodeError:
        return ""


def diarize(audio_bytes: bytes, text: str) -> str:
    """Step 2 — Pyannote speaker diarization (prepends SPEAKER_XX labels)."""
    if not DIARIZATION_AVAILABLE or not text:
        return text
    try:
        samples = decode_audio(audio_bytes)
        waveform = torch.tensor(samples).unsqueeze(0)
        diarization = diarizer({"waveform": waveform, "sample_rate": 16000})
        # Map each turn to its label; use the first turn's speaker as prefix
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            return f"[{speaker}] {text}"
    except Exception:
        pass
    return text


def resolve_coreferences(text: str) -> str:
    """Step 3 — Coreferee coreference resolution (pronoun → noun)."""
    if not COREF_AVAILABLE or not text:
        return text
    try:
        doc = nlp(text)
        resolved = []
        for token in doc:
            repres = doc._.coref_chains.resolve(token)
            if repres:
                resolved.append(" ".join(t.text for t in repres) + token.whitespace_)
            else:
                resolved.append(token.text_with_ws)
        return "".join(resolved).strip()
    except Exception:
        return text


def compute_drift(text: str, context_vec) -> float | None:
    """Step 4 — Sentence-Transformer cosine drift score."""
    if len(text.strip()) < 10:
        return None
    chunk_vec = embedder.encode([text])
    similarity = cosine_similarity(context_vec, chunk_vec)[0][0]
    return float(1 - similarity)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    config = await ws.receive_json()
    context: str = config["context"]
    tolerance: float = config["tolerance"]
    context_vec = embedder.encode([context])

    drift_history: list[float] = []
    webm_header = None  # first chunk from MediaRecorder contains WebM init segment

    loop = asyncio.get_event_loop()

    def full_pipeline(buf: bytes) -> dict:
        raw_text = transcribe(buf)
        if not raw_text or len(raw_text.strip()) < 5:
            return {}
        labelled = diarize(buf, raw_text)
        resolved = resolve_coreferences(labelled)
        drift = compute_drift(resolved, context_vec)
        if drift is None:
            return {}
        return {"text": resolved, "drift": drift}

    try:
        while True:
            # Each chunk is exactly one 15-second MediaRecorder timeslice
            audio_chunk = await ws.receive_bytes()

            if webm_header is None:
                # First chunk is self-contained (init segment + first 15s audio)
                webm_header = audio_chunk
                full_audio = audio_chunk
            else:
                # Subsequent chunks need the init segment prepended to be decodable
                full_audio = webm_header + audio_chunk

            result = await loop.run_in_executor(None, full_pipeline, full_audio)

            if not result:
                continue

            drift_history.append(result["drift"])
            if len(drift_history) > 3:
                drift_history.pop(0)

            drift_avg = sum(drift_history) / len(drift_history)

            await ws.send_json({
                "text": result["text"],
                "drift": drift_avg,
                "alert": bool(drift_avg > tolerance),
            })

    except WebSocketDisconnect:
        print("Client disconnected")
