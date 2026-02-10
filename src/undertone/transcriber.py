"""Speech-to-text transcription via Groq cloud and local Whisper."""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time

import httpx

log = logging.getLogger("undertone")

# HTTP status codes that are worth retrying
_RETRYABLE_STATUSES = {429, 500, 502, 503}
_MAX_RETRIES = 2
_BACKOFF_SCHEDULE = (0.3, 0.6)


class GroqTranscriber:
    """Transcribe audio via the Groq cloud API."""

    API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        language: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self.prompt = prompt
        self._client = httpx.Client(timeout=30.0)

    def transcribe(self, audio_buf: io.BytesIO) -> str:
        """Transcribe audio with automatic retry on transient failures."""
        if not self.api_key:
            raise ValueError("No Groq API key configured")

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                audio_buf.seek(0)
                files = {"file": ("audio.wav", audio_buf, "audio/wav")}
                data: dict[str, str] = {"model": self.model}
                if self.language:
                    data["language"] = self.language
                if self.prompt:
                    data["prompt"] = self.prompt

                resp = self._client.post(
                    self.API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    data=data,
                )

                if resp.status_code in _RETRYABLE_STATUSES and attempt < _MAX_RETRIES:
                    wait = _BACKOFF_SCHEDULE[attempt]
                    log.warning(
                        f"Groq returned {resp.status_code}, retrying in {wait}s "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return str(resp.json()["text"]).strip()

            except httpx.TimeoutException as e:
                last_exc = e
                if attempt < _MAX_RETRIES:
                    wait = _BACKOFF_SCHEDULE[attempt]
                    log.warning(
                        f"Groq timed out, retrying in {wait}s "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                    )
                    time.sleep(wait)
                    continue
                raise

            except httpx.HTTPStatusError:
                raise  # 4xx (non-429) — don't retry

            except Exception as e:
                last_exc = e
                if attempt < _MAX_RETRIES:
                    wait = _BACKOFF_SCHEDULE[attempt]
                    log.warning(
                        f"Groq error ({e}), retrying in {wait}s "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                    )
                    time.sleep(wait)
                    continue
                raise

        raise last_exc  # type: ignore[misc]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


class LocalTranscriber:
    """Transcribe audio locally using faster-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None  # lazy-loaded

    def preload(self) -> None:
        """Load the Whisper model."""
        if self._model is not None:
            return
        log.info(f"Loading local Whisper model '{self.model_size}'...")
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        log.info("Local Whisper model loaded")

    def transcribe(self, audio_buf: io.BytesIO) -> str:
        """Transcribe audio from a WAV BytesIO buffer."""
        if self._model is None:
            self.preload()

        audio_buf.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_buf.read())
            tmp_path = tmp.name

        try:
            assert self._model is not None
            segments, _ = self._model.transcribe(tmp_path, vad_filter=True)
            return " ".join(seg.text for seg in segments).strip()
        finally:
            os.unlink(tmp_path)


def route_transcription(
    audio_buf: io.BytesIO,
    groq: GroqTranscriber,
    local: LocalTranscriber,
    config: dict,
) -> tuple[str, str]:
    """Try Groq first, fall back to local on any failure."""
    primary = config.get("stt", {}).get("primary", "groq")

    if primary == "groq" and groq.api_key:
        try:
            text = groq.transcribe(audio_buf)
            log.info(f'[Groq] "{text}"')
            return text, "groq"
        except Exception as e:
            log.warning(f"Groq failed ({e}), falling back to local")
            audio_buf.seek(0)

    text = local.transcribe(audio_buf)
    log.info(f'[Local] "{text}"')
    return text, "local"
