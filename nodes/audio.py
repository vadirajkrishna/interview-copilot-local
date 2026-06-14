import asyncio
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from config import BASE_DIR, HF_WHISPER_MODEL, STT_BACKEND, WHISPER_MODEL
from state import CoachState

SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
OVERLAP_SECONDS = 0.5
CHUNK_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)
OVERLAP_SAMPLES = int(OVERLAP_SECONDS * SAMPLE_RATE)
QUEUE_MAX_SIZE = 10
SILENCE_RMS_THRESHOLD = 0.003

executor = ThreadPoolExecutor(max_workers=2)
_asr_pipeline = None


async def audio_node(state: CoachState) -> CoachState:
    if "audio_queue" not in state:
        state["audio_queue"] = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    return state


async def enqueue_audio(audio_queue: asyncio.Queue, chunk: Any) -> None:
    await audio_queue.put(chunk)


def get_input_device() -> int | None:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception:
        return None

    for index, device in enumerate(devices):
        name = str(device.get("name", ""))
        max_inputs = int(device.get("max_input_channels", 0))
        if "BlackHole" in name and max_inputs > 0:
            return index
    return None


class LiveAudioTranscriber:
    def __init__(self, model: str = WHISPER_MODEL):
        self.model = model
        self.audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        self.stop_event = asyncio.Event()
        self.capture_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.capture_task and not self.capture_task.done():
            await self.stop()
        self.stop_event = asyncio.Event()
        self.audio_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        device_index = get_input_device()
        self.capture_task = asyncio.create_task(
            capture_audio(
                audio_queue=self.audio_queue,
                stop_event=self.stop_event,
                device_index=device_index,
            )
        )

    async def stop(self) -> None:
        self.stop_event.set()
        if self.capture_task:
            await self.capture_task
            self.capture_task = None

    async def transcript_stream(self):
        transcript = ""
        while not self.stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                text = await transcribe_chunk(chunk, model=self.model)
            except Exception as exc:
                text = f"[live transcription error: {exc}]"
            if text:
                transcript = merge_chunk_text(transcript, text)
                yield transcript


async def capture_audio(
    audio_queue: asyncio.Queue[np.ndarray],
    stop_event: asyncio.Event,
    device_index: int | None = None,
) -> None:
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError(
            "sounddevice is required for live audio capture. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc

    loop = asyncio.get_running_loop()
    buffer: list[float] = []

    def callback(indata, frames, time, status):
        if status:
            return

        buffer.extend(indata[:, 0].tolist())
        while len(buffer) >= CHUNK_SAMPLES:
            chunk = np.array(buffer[:CHUNK_SAMPLES], dtype=np.float32)
            buffer[:] = buffer[CHUNK_SAMPLES - OVERLAP_SAMPLES :]
            if is_silent(chunk):
                continue
            loop.call_soon_threadsafe(enqueue_chunk_nowait, audio_queue, chunk)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device_index,
        callback=callback,
        blocksize=1024,
    ):
        while not stop_event.is_set():
            await asyncio.sleep(0.1)


def enqueue_chunk_nowait(audio_queue: asyncio.Queue[np.ndarray], chunk: np.ndarray) -> None:
    if audio_queue.full():
        try:
            audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    audio_queue.put_nowait(chunk)


async def transcribe_chunk(audio: np.ndarray, model: str = WHISPER_MODEL) -> str:
    if audio.size == 0 or is_silent(audio):
        return ""

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: _transcribe_audio_array_sync(
            audio,
            model,
            {
                "language": "en",
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 1.8,
                "logprob_threshold": -0.6,
                "no_speech_threshold": 0.35,
            },
        ),
    )
    return "" if is_repetitive_hallucination(result) else result


def is_silent(audio: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
    return rms < threshold


def merge_chunk_text(existing: str, incoming: str) -> str:
    existing = " ".join(existing.split()).strip()
    incoming = " ".join(incoming.split()).strip()
    if not existing:
        return incoming
    if not incoming:
        return existing
    if incoming.lower().startswith(existing.lower()):
        return incoming
    if existing.lower().endswith(incoming.lower()):
        return existing

    existing_lower = existing.lower()
    incoming_lower = incoming.lower()
    max_overlap = min(len(existing), len(incoming), 120)
    for size in range(max_overlap, 4, -1):
        if existing_lower.endswith(incoming_lower[:size]):
            return f"{existing}{incoming[size:]}".strip()
    return f"{existing} {incoming}".strip()


def is_repetitive_hallucination(text: str) -> bool:
    import re

    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 8:
        return False
    unique_words = set(words)
    if len(unique_words) <= 2:
        return True
    most_common = max(words.count(word) for word in unique_words)
    return most_common / len(words) >= 0.65


async def transcribe_audio_file(
    audio_path: str,
    model: str = WHISPER_MODEL,
    backend: str | None = None,
    hf_model: str = HF_WHISPER_MODEL,
) -> str:
    if not audio_path:
        return ""

    return await asyncio.to_thread(_transcribe_audio_file_sync, audio_path, model, backend, hf_model)


async def transcribe_audio_array(
    sample_rate: int,
    audio: np.ndarray,
    model: str = WHISPER_MODEL,
    backend: str | None = None,
    hf_model: str = HF_WHISPER_MODEL,
    **decode_options: Any,
) -> str:
    if audio.size == 0:
        return ""

    waveform = prepare_audio_array(sample_rate, audio)
    return await asyncio.to_thread(
        _transcribe_audio_array_sync,
        waveform,
        model,
        decode_options,
        backend,
        hf_model,
    )


def _transcribe_audio_file_sync(
    audio_path: str,
    model: str,
    backend: str | None = None,
    hf_model: str = HF_WHISPER_MODEL,
) -> str:
    if (backend or STT_BACKEND) == "transformers":
        return _transcribe_with_transformers(audio_path, hf_model)

    try:
        ensure_ffmpeg_on_path()
        import mlx_whisper

        result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model)
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()
    except Exception as exc:
        return f"[transcription unavailable: {exc}]"


def _transcribe_audio_array_sync(
    waveform: np.ndarray,
    model: str,
    decode_options: dict[str, Any] | None = None,
    backend: str | None = None,
    hf_model: str = HF_WHISPER_MODEL,
) -> str:
    if (backend or STT_BACKEND) == "transformers":
        return _transcribe_with_transformers(
            {"array": waveform.astype(np.float32), "sampling_rate": 16000},
            hf_model,
        )

    try:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            waveform,
            path_or_hf_repo=model,
            verbose=False,
            **(decode_options or {}),
        )
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()
    except Exception as exc:
        return f"[transcription unavailable: {exc}]"


def _transcribe_with_transformers(audio_input: Any, model: str) -> str:
    try:
        pipeline = get_asr_pipeline(model)
        result = pipeline(audio_input, generate_kwargs={"language": "english", "task": "transcribe"})
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()
    except Exception as exc:
        return f"[transcription unavailable: {exc}]"


def get_asr_pipeline(model: str):
    global _asr_pipeline
    if _asr_pipeline is None:
        from transformers import pipeline

        _asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
        )
    return _asr_pipeline


def prepare_audio_array(sample_rate: int, audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)

    if sample_rate != 16000:
        from scipy.signal import resample_poly

        gcd = np.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // gcd, sample_rate // gcd).astype(np.float32)

    return audio


def ensure_ffmpeg_on_path() -> None:
    if shutil.which("ffmpeg"):
        return

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = BASE_DIR / ".runtime" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        shim = bin_dir / "ffmpeg"
        if not shim.exists():
            shim.symlink_to(ffmpeg)
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is required for audio decoding. Install it with `brew install ffmpeg` "
            "or `python3 -m pip install imageio-ffmpeg`."
        ) from exc
