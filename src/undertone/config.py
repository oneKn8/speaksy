"""Configuration management for Undertone."""

import os
from pathlib import Path

import yaml

# XDG config directory
CONFIG_DIR = Path.home() / ".config" / "undertone"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
ENV_FILE = CONFIG_DIR / ".env"

DEFAULT_CONFIG = {
    "stt": {
        "primary": "groq",
        "groq_model": "whisper-large-v3-turbo",
        "local_model": "base",
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


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def ensure_config_dir():
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


def save_config(config: dict):
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


def save_api_key(api_key: str):
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


def get_hotkeys() -> tuple:
    """Get current hotkey settings."""
    config = load_config()
    hotkeys = config.get("hotkeys", {})
    return (
        hotkeys.get("push_to_talk", "Key.ctrl_r"),
        hotkeys.get("toggle", "Key.f8"),
    )


def set_hotkeys(push_to_talk: str, toggle: str):
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


def set_privacy_mode(mode: str):
    """Set privacy mode (cloud or local)."""
    config = load_config()
    config["stt"]["primary"] = "local" if mode == "local" else "groq"
    save_config(config)


def get_cleanup_enabled() -> bool:
    """Check if text cleanup is enabled."""
    config = load_config()
    return config.get("cleanup", {}).get("enabled", True)


def set_cleanup_enabled(enabled: bool):
    """Enable or disable text cleanup."""
    config = load_config()
    if "cleanup" not in config:
        config["cleanup"] = {}
    config["cleanup"]["enabled"] = enabled
    save_config(config)


def get_cleanup_llm_enabled() -> bool:
    """Check if LLM-based text cleanup is enabled."""
    config = load_config()
    return config.get("cleanup", {}).get("llm_enabled", True)


def set_cleanup_llm_enabled(enabled: bool):
    """Enable or disable LLM-based text cleanup."""
    config = load_config()
    if "cleanup" not in config:
        config["cleanup"] = {}
    config["cleanup"]["llm_enabled"] = enabled
    save_config(config)
