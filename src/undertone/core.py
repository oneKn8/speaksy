"""Core voice typing engine for Undertone."""

import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque

import httpx
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw
from pynput import keyboard

try:
    import pystray
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False

log = logging.getLogger("undertone")


# ---------------------------------------------------------------------------
# Audio Recorder
# ---------------------------------------------------------------------------


class AudioRecorder:
    """Captures microphone audio with a rolling pre-buffer."""

    def __init__(self, sample_rate=16000, channels=1, pre_buffer_sec=0.5):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 1024
        pre_buffer_chunks = int(sample_rate * pre_buffer_sec / self.chunk_size)
        self.pre_buffer = deque(maxlen=max(pre_buffer_chunks, 1))
        self.recording_chunks = []
        self.is_recording = False
        self.stream = None
        self._lock = threading.Lock()

    def open(self):
        """Start the always-on audio input stream for pre-buffering."""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self.stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata.copy()
        with self._lock:
            if self.is_recording:
                self.recording_chunks.append(chunk)
            else:
                self.pre_buffer.append(chunk)

    def start_recording(self):
        """Begin capturing audio, including the pre-buffer."""
        with self._lock:
            self.recording_chunks = list(self.pre_buffer)
            self.pre_buffer.clear()
            self.is_recording = True
        log.info("Recording started")

    def stop_recording(self):
        """Stop capturing and return audio as an in-memory WAV BytesIO."""
        with self._lock:
            self.is_recording = False
            chunks = self.recording_chunks
            self.recording_chunks = []

        if not chunks:
            log.warning("No audio captured")
            return None

        audio_data = np.concatenate(chunks, axis=0)
        duration = len(audio_data) / self.sample_rate
        log.info(f"Captured {duration:.1f}s of audio")

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        buf.seek(0)
        return buf

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# ---------------------------------------------------------------------------
# Transcribers
# ---------------------------------------------------------------------------


class GroqTranscriber:
    """Transcribe audio via the Groq cloud API."""

    API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self, api_key, model="whisper-large-v3-turbo", language=None):
        self.api_key = api_key
        self.model = model
        self.language = language

    def transcribe(self, audio_buf):
        if not self.api_key:
            raise ValueError("No Groq API key configured")

        audio_buf.seek(0)
        files = {"file": ("audio.wav", audio_buf, "audio/wav")}
        data = {"model": self.model}
        if self.language:
            data["language"] = self.language

        resp = httpx.post(
            self.API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files=files,
            data=data,
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["text"].strip()


class LocalTranscriber:
    """Transcribe audio locally using faster-whisper."""

    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def preload(self):
        """Load the Whisper model."""
        if self._model is not None:
            return
        log.info(f"Loading local Whisper model '{self.model_size}'...")
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        log.info("Local Whisper model loaded")

    def transcribe(self, audio_buf):
        if self._model is None:
            self.preload()

        audio_buf.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_buf.read())
            tmp_path = tmp.name

        try:
            segments, _ = self._model.transcribe(tmp_path, vad_filter=True)
            return " ".join(seg.text for seg in segments).strip()
        finally:
            os.unlink(tmp_path)


def route_transcription(audio_buf, groq, local, config):
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


# ---------------------------------------------------------------------------
# Text Cleanup
# ---------------------------------------------------------------------------

FILLER_PATTERNS = [
    r"\b(um+|uh+|er+|ah+)\b",
    r"\b(like,?\s+)(?=\w)",
    r"\b(you know,?\s*)",
    r"\b(basically,?\s*)",
    r"\b(actually,?\s*)(?![\w])",
    r"\b(so,?\s+)(?=[a-z])",
    r"\b(i mean,?\s*)",
    r"\b(kind of|kinda)\s+",
    r"\b(sort of|sorta)\s+",
]

CLEANUP_SYSTEM_PROMPT = (
    "You are a text formatter. Fix grammar, punctuation, and capitalization "
    "in the following transcribed speech. Rules:\n"
    "- Fix grammar errors\n"
    "- Add proper punctuation (periods, commas, question marks)\n"
    "- Capitalize sentences and proper nouns\n"
    "- Remove filler words (um, uh, like, you know, basically, etc.)\n"
    "- Do NOT add, remove, or rephrase content beyond fixing errors\n"
    "- Do NOT add explanations or commentary\n"
    "- Return ONLY the corrected text, nothing else"
)


class TextCleaner:
    """Clean up transcribed text using regex + optional LLM grammar fix."""

    CHAT_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key=None, model=None, llm_enabled=False):
        self.api_key = api_key
        self.model = model or "llama-3.1-8b-instant"
        self.llm_enabled = llm_enabled and bool(api_key)

    def _regex_clean(self, text):
        """Stage 1: Fast regex-based filler removal."""
        cleaned = text
        for pattern in FILLER_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."

        return cleaned

    def _llm_clean(self, text):
        """Stage 2: LLM-based grammar and punctuation fix via Groq."""
        try:
            resp = httpx.post(
                self.CHAT_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"].strip()
            # Guard against LLM returning empty or wildly different length text
            if result and 0.3 < len(result) / max(len(text), 1) < 3.0:
                return result
            log.warning("[LLM Cleanup] Result looks suspicious, using regex fallback")
            return None
        except Exception as e:
            log.warning(f"[LLM Cleanup] Failed ({e}), using regex fallback")
            return None

    def clean(self, text):
        if not text:
            return text

        original = text

        if self.llm_enabled:
            llm_result = self._llm_clean(text)
            if llm_result:
                if llm_result != original:
                    log.info(f'[LLM Cleanup] "{original}" -> "{llm_result}"')
                return llm_result

        # Fallback: regex-only cleanup
        cleaned = self._regex_clean(text)
        if cleaned != original:
            log.info(f'[Regex Cleanup] "{original}" -> "{cleaned}"')
        return cleaned


# ---------------------------------------------------------------------------
# Session Detection & Text Injection
# ---------------------------------------------------------------------------


def detect_session():
    """Detect display server: 'wayland' or 'x11'."""
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session in ("wayland", "x11"):
        return session
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "x11"


def _detect_tools():
    """Pick the right clipboard and key simulation tools for the session."""
    session = detect_session()

    if session == "wayland":
        # Clipboard: prefer wl-copy, fall back to xclip (works via XWayland)
        if shutil.which("wl-copy"):
            clip_copy = ["wl-copy"]
            clip_paste = ["wl-paste"]
        else:
            clip_copy = ["xclip", "-sel", "clip"]
            clip_paste = ["xclip", "-sel", "clip", "-o"]

        # Key simulation: prefer wtype (fast), then ydotool, then xdotool
        if shutil.which("wtype"):
            key_tool = "wtype"
        elif shutil.which("ydotool"):
            key_tool = "ydotool"
        else:
            key_tool = "xdotool"
    else:
        clip_copy = ["xclip", "-sel", "clip"]
        clip_paste = ["xclip", "-sel", "clip", "-o"]
        key_tool = "xdotool"

    return clip_copy, clip_paste, key_tool


# Cache detected tools at module level (session doesn't change mid-run)
_CLIP_COPY, _CLIP_PASTE, _KEY_TOOL = _detect_tools()


def _simulate_paste():
    """Simulate Ctrl+V using the appropriate tool for the session."""
    if _KEY_TOOL == "wtype":
        subprocess.run(["wtype", "-M", "ctrl", "v", "-m", "ctrl"], timeout=2)
    elif _KEY_TOOL == "ydotool":
        # Raw keycodes: 29=KEY_LEFTCTRL, 47=KEY_V
        subprocess.run(
            ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"], timeout=2
        )
    else:
        subprocess.run(["xdotool", "key", "ctrl+v"], timeout=2)


def inject_text(text, restore_clipboard=True):
    """Type text at cursor position via clipboard paste."""
    if not text:
        return

    old_clipboard = None
    if restore_clipboard:
        try:
            result = subprocess.run(
                _CLIP_PASTE, capture_output=True, timeout=2,
            )
            if result.returncode == 0:
                old_clipboard = result.stdout
        except Exception:
            pass

    proc = subprocess.Popen(_CLIP_COPY, stdin=subprocess.PIPE)
    proc.communicate(text.encode("utf-8"))

    time.sleep(0.05)
    _simulate_paste()
    time.sleep(0.1)

    if restore_clipboard and old_clipboard is not None:
        time.sleep(0.1)
        proc = subprocess.Popen(_CLIP_COPY, stdin=subprocess.PIPE)
        proc.communicate(old_clipboard)


# ---------------------------------------------------------------------------
# Tray Icon
# ---------------------------------------------------------------------------

TRAY_COLORS = {
    "ready": (76, 175, 80, 255),
    "recording": (244, 67, 54, 255),
    "processing": (255, 193, 7, 255),
    "fallback": (255, 152, 0, 255),
}

TRAY_TITLES = {
    "ready": "Undertone - Ready",
    "recording": "Undertone - Recording...",
    "processing": "Undertone - Transcribing...",
    "fallback": "Undertone - Local Mode",
}


def _make_circle_icon(color, size=64):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, size - 4, size - 4], fill=color)
    return img


class TrayManager:
    """System tray icon showing recording state."""

    def __init__(self, on_quit):
        self.on_quit = on_quit
        self._icon = None

    def start(self):
        menu = pystray.Menu(
            pystray.MenuItem("Undertone", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: self._quit()),
        )
        self._icon = pystray.Icon(
            "undertone",
            _make_circle_icon(TRAY_COLORS["ready"]),
            TRAY_TITLES["ready"],
            menu,
        )
        self._icon.run()

    def set_state(self, state):
        if self._icon:
            self._icon.icon = _make_circle_icon(
                TRAY_COLORS.get(state, TRAY_COLORS["ready"])
            )
            self._icon.title = TRAY_TITLES.get(state, "Undertone")

    def _quit(self):
        if self._icon:
            self._icon.stop()
        self.on_quit()


# ---------------------------------------------------------------------------
# Hotkey Manager
# ---------------------------------------------------------------------------


def _parse_key(key_str):
    """Parse 'Key.ctrl_r' or 'Key.f8' to a pynput key object."""
    if key_str.startswith("Key."):
        attr = key_str[4:]
        return getattr(keyboard.Key, attr)
    return keyboard.KeyCode.from_char(key_str)


class HotkeyManager:
    """Global hotkey listener for push-to-talk and toggle modes."""

    def __init__(self, push_to_talk_key, toggle_key, on_start, on_stop):
        self.ptt_key = _parse_key(push_to_talk_key)
        self.toggle_key = _parse_key(toggle_key)
        self.on_start = on_start
        self.on_stop = on_stop
        self._ptt_held = False
        self._toggle_active = False
        self._listener = None

    def start(self):
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        log.info(
            f"Hotkeys active: hold {self.ptt_key} (push-to-talk), "
            f"press {self.toggle_key} (toggle)"
        )

    def _on_press(self, key):
        if key == self.ptt_key and not self._ptt_held:
            self._ptt_held = True
            if not self._toggle_active:
                self.on_start()
        elif key == self.toggle_key:
            if self._toggle_active:
                self._toggle_active = False
                self.on_stop()
            else:
                self._toggle_active = True
                self.on_start()

    def _on_release(self, key):
        if key == self.ptt_key and self._ptt_held:
            self._ptt_held = False
            if not self._toggle_active:
                self.on_stop()

    def stop(self):
        if self._listener:
            self._listener.stop()


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class UndertoneEngine:
    """Orchestrates recording, transcription, and text injection."""

    def __init__(self, config, api_key=None):
        self.config = config
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._recording = False
        self._transcribing = False
        self._using_fallback = False

        audio_cfg = config.get("audio", {})
        self.recorder = AudioRecorder(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            channels=audio_cfg.get("channels", 1),
            pre_buffer_sec=audio_cfg.get("pre_buffer_seconds", 0.5),
        )

        stt_cfg = config.get("stt", {})
        self.groq = GroqTranscriber(
            api_key=self.api_key,
            model=stt_cfg.get("groq_model", "whisper-large-v3-turbo"),
            language=stt_cfg.get("language"),
        )
        self.local = LocalTranscriber(
            model_size=stt_cfg.get("local_model", "base"),
            device=stt_cfg.get("local_device", "cpu"),
            compute_type=stt_cfg.get("local_compute_type", "int8"),
        )

        cleanup_cfg = config.get("cleanup", {})
        self.cleaner = None
        if cleanup_cfg.get("enabled", True):
            self.cleaner = TextCleaner(
                api_key=self.api_key,
                model=cleanup_cfg.get("model", "llama-3.1-8b-instant"),
                llm_enabled=cleanup_cfg.get("llm_enabled", True),
            )

        tray_cfg = config.get("tray", {})
        self.tray = None
        if tray_cfg.get("enabled", True) and HAS_TRAY:
            self.tray = TrayManager(on_quit=self.shutdown)

        hotkey_cfg = config.get("hotkeys", {})
        self.hotkeys = HotkeyManager(
            push_to_talk_key=hotkey_cfg.get("push_to_talk", "Key.ctrl_r"),
            toggle_key=hotkey_cfg.get("toggle", "Key.f8"),
            on_start=self._on_record_start,
            on_stop=self._on_record_stop,
        )

    def _on_record_start(self):
        if self._recording or self._transcribing:
            return
        self._recording = True
        self.recorder.start_recording()
        if self.tray:
            self.tray.set_state("recording")

    def _on_record_stop(self):
        if not self._recording:
            return
        self._recording = False
        audio_buf = self.recorder.stop_recording()
        if audio_buf is None:
            if self.tray:
                self.tray.set_state("ready")
            return

        self._transcribing = True
        if self.tray:
            self.tray.set_state("processing")
        threading.Thread(
            target=self._transcribe_and_type, args=(audio_buf,), daemon=True
        ).start()

    def _transcribe_and_type(self, audio_buf):
        try:
            text, source = route_transcription(
                audio_buf, self.groq, self.local, self.config
            )
            self._using_fallback = source == "local"

            if text:
                if self.cleaner:
                    text = self.cleaner.clean(text)

                text_cfg = self.config.get("text_injection", {})
                inject_text(
                    text,
                    restore_clipboard=text_cfg.get("restore_clipboard", True),
                )
        except Exception as e:
            log.error(f"Transcription failed: {e}")
        finally:
            self._transcribing = False
            if self.tray:
                state = "fallback" if self._using_fallback else "ready"
                self.tray.set_state(state)

    def run(self):
        """Start the engine (blocking)."""
        self.recorder.open()

        # Preload local model in background
        threading.Thread(target=self.local.preload, daemon=True).start()

        self.hotkeys.start()

        session = detect_session()
        log.info(f"Session: {session} | clipboard: {_CLIP_COPY[0]} | keys: {_KEY_TOOL}")

        api_status = "configured" if self.api_key else "NOT SET (local only)"
        log.info(f"Undertone ready. API key: {api_status}")

        hotkey_cfg = self.config.get("hotkeys", {})
        ptt = hotkey_cfg.get("push_to_talk", "Key.ctrl_r")
        toggle = hotkey_cfg.get("toggle", "Key.f8")
        log.info(f"Hold {ptt} (push-to-talk) or press {toggle} (toggle)")

        if self.tray:
            self.tray.start()
        else:
            try:
                signal.pause()
            except KeyboardInterrupt:
                pass

    def shutdown(self):
        """Stop the engine."""
        log.info("Shutting down...")
        self.hotkeys.stop()
        self.recorder.close()
        sys.exit(0)
