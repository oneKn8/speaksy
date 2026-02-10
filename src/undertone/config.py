"""Configuration management for Undertone."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

import yaml  # type: ignore[import-untyped]

# XDG config directory
CONFIG_DIR = Path.home() / ".config" / "undertone"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
ENV_FILE = CONFIG_DIR / ".env"


# ---------------------------------------------------------------------------
# TypedDict schemas for config structure
# ---------------------------------------------------------------------------


class STTConfig(TypedDict, total=False):
    primary: str
    groq_model: str
    local_model: str
    local_device: str
    local_compute_type: str
    language: str


class CleanupConfig(TypedDict, total=False):
    enabled: bool
    llm_enabled: bool
    model: str


class AudioConfig(TypedDict, total=False):
    sample_rate: int
    channels: int
    sound_feedback: bool
    pre_buffer_seconds: float


class HotkeyConfig(TypedDict, total=False):
    push_to_talk: str
    toggle: str


class TextInjectionConfig(TypedDict, total=False):
    method: str
    restore_clipboard: bool


class TrayConfig(TypedDict, total=False):
    enabled: bool


class UndertoneConfig(TypedDict, total=False):
    stt: STTConfig
    cleanup: CleanupConfig
    audio: AudioConfig
    hotkeys: HotkeyConfig
    text_injection: TextInjectionConfig
    tray: TrayConfig


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "stt": {
        "primary": "groq",
        "groq_model": "whisper-large-v3-turbo",
        "local_model": "distil-large-v3",
        "local_device": "cpu",
        "local_compute_type": "int8",
        "language": "en",
    },
    "cleanup": {
        "enabled": True,
        "llm_enabled": True,
        "model": "llama-3.1-8b-instant",
    },
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "sound_feedback": True,
        "pre_buffer_seconds": 0.5,
    },
    "hotkeys": {
        "push_to_talk": "Key.ctrl_r",
        "toggle": "Key.f8",
    },
    "text_injection": {
        "method": "clipboard",
        "restore_clipboard": True,
    },
    "tray": {
        "enabled": True,
    },
}


# ---------------------------------------------------------------------------
# Config functions
# ---------------------------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def ensure_config_dir() -> None:
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def config_exists() -> bool:
    """Check if config file exists."""
    return CONFIG_FILE.exists()


def load_config() -> dict:
    """Load config from YAML, merge with defaults."""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            user_config = yaml.safe_load(f) or {}
        config = deep_merge(DEFAULT_CONFIG, user_config)
    return config


def save_config(config: dict) -> None:
    """Save config to YAML file."""
    ensure_config_dir()
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_api_key() -> str:
    """Get API key from .env file."""
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return os.getenv("GROQ_API_KEY", "")


def save_api_key(api_key: str) -> None:
    """Save API key to .env file."""
    ensure_config_dir()
    with open(ENV_FILE, "w") as f:
        f.write(f"GROQ_API_KEY={api_key}\n")
    # Set restrictive permissions
    os.chmod(ENV_FILE, 0o600)


def api_key_exists() -> bool:
    """Check if API key is configured."""
    return bool(get_api_key())


def is_configured() -> bool:
    """Check if undertone is fully configured."""
    return config_exists() and api_key_exists()


def get_hotkeys() -> tuple[str, str]:
    """Get current hotkey settings."""
    config = load_config()
    hotkeys = config.get("hotkeys", {})
    return (
        hotkeys.get("push_to_talk", "Key.ctrl_r"),
        hotkeys.get("toggle", "Key.f8"),
    )


def set_hotkeys(push_to_talk: str, toggle: str) -> None:
    """Update hotkey settings."""
    config = load_config()
    config["hotkeys"]["push_to_talk"] = push_to_talk
    config["hotkeys"]["toggle"] = toggle
    save_config(config)


def get_privacy_mode() -> str:
    """Get current privacy mode (cloud or local)."""
    config = load_config()
    primary = config.get("stt", {}).get("primary", "groq")
    return "local" if primary == "local" else "cloud"


def set_privacy_mode(mode: str) -> None:
    """Set privacy mode (cloud or local)."""
    config = load_config()
    config["stt"]["primary"] = "local" if mode == "local" else "groq"
    save_config(config)


def get_cleanup_enabled() -> bool:
    """Check if text cleanup is enabled."""
    config = load_config()
    return bool(config.get("cleanup", {}).get("enabled", True))


def set_cleanup_enabled(enabled: bool) -> None:
    """Enable or disable text cleanup."""
    config = load_config()
    if "cleanup" not in config:
        config["cleanup"] = {}
    config["cleanup"]["enabled"] = enabled
    save_config(config)


def get_cleanup_llm_enabled() -> bool:
    """Check if LLM-based text cleanup is enabled."""
    config = load_config()
    return bool(config.get("cleanup", {}).get("llm_enabled", True))


def set_cleanup_llm_enabled(enabled: bool) -> None:
    """Enable or disable LLM-based text cleanup."""
    config = load_config()
    if "cleanup" not in config:
        config["cleanup"] = {}
    config["cleanup"]["llm_enabled"] = enabled
    save_config(config)


def get_sound_feedback() -> bool:
    """Check if sound feedback is enabled."""
    config = load_config()
    return bool(config.get("audio", {}).get("sound_feedback", True))


def set_sound_feedback(enabled: bool) -> None:
    """Enable or disable sound feedback."""
    config = load_config()
    if "audio" not in config:
        config["audio"] = {}
    config["audio"]["sound_feedback"] = enabled
    save_config(config)


def get_language() -> str:
    """Get current STT language."""
    config = load_config()
    return str(config.get("stt", {}).get("language", "en"))


def set_language(language: str) -> None:
    """Set STT language."""
    config = load_config()
    if "stt" not in config:
        config["stt"] = {}
    config["stt"]["language"] = language
    save_config(config)


def get_whisper_prompt() -> str:
    """Get the Whisper prompt hint (helps with accents/vocabulary)."""
    config = load_config()
    return str(config.get("stt", {}).get("prompt", ""))


def set_whisper_prompt(prompt: str) -> None:
    """Set the Whisper prompt hint."""
    config = load_config()
    if "stt" not in config:
        config["stt"] = {}
    config["stt"]["prompt"] = prompt
    save_config(config)
