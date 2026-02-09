<p align="center">
  <img src="https://img.shields.io/badge/undertone-voice%20typing-blueviolet?style=for-the-badge&logo=microphone" alt="undertone">
</p>

<h1 align="center">undertone</h1>

<p align="center">
  <strong>talk it. type it. ship it.</strong>
</p>

<p align="center">
  <a href="https://github.com/oneKn8/undertone/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"></a>
  <a href="https://github.com/oneKn8/undertone"><img src="https://img.shields.io/badge/platform-Linux-orange.svg" alt="Platform"></a>
  <a href="https://console.groq.com"><img src="https://img.shields.io/badge/powered%20by-Groq-ff6600.svg" alt="Groq"></a>
</p>

<p align="center">
  <em>Voice typing for Linux that actually works.<br>Hold a key, speak, release -- your words appear wherever you're typing.</em>
</p>

---

## Demo

```
$ undertone

  ╭────────────────────────────────────────╮
  │  UNDERTONE                             │
  │  talk it. type it. ship it.            │
  ╰────────────────────────────────────────╯

  Status: vibing
  Hotkeys: Right Ctrl (hold) | F8 (toggle)

undertone> _
```

---

## Quick Start

```bash
# Install
pipx install undertone

# Run (interactive setup on first launch)
undertone

# Upgrade to latest version
pipx upgrade undertone
```

That's it. 30 seconds to voice typing.

---

## Features

| | Feature | Description |
|---|---------|-------------|
| **Speed** | < 1 second latency | Groq's Whisper API is blazing fast |
| **Smart** | Grammar + filler fix | LLM fixes grammar, punctuation, removes filler words |
| **Free** | No credit card | Groq's free tier is generous |
| **Offline** | Local fallback | Works without internet via faster-whisper |
| **Private** | Privacy mode | Keep voice 100% on your machine |
| **Auto** | Runs on login | Always ready when you are |

---

## How It Works

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Hold Key   │ -> │   Speak     │ -> │  Release    │ -> │ Text Appears│
│  (Right Ctrl)    │  naturally  │    │   key       │    │  at cursor  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                            |
                            v
                   ┌─────────────────┐
                   │  Groq Whisper   │
                   │ + LLM cleanup   │
                   └─────────────────┘
```

1. Press hotkey (Right Ctrl = hold, F8 = toggle)
2. Speak naturally
3. Release -- text appears in < 1 second

Works everywhere: browser, terminal, IDE, Slack, Discord, anywhere you type.

---

## Commands

Run `undertone` to open the interactive CLI:

| Command | Description |
|---------|-------------|
| `/setup` | Configure API key & hotkeys |
| `/start` | Start voice typing |
| `/stop` | Take a break |
| `/status` | Check the vibe |
| `/logs` | View receipts |
| `/config` | Tweak settings |
| `/help` | Get backup |
| `/quit` | Peace out |

---

## Requirements

- **OS:** Linux (X11, Wayland, or XWayland)
- **Python:** 3.10+
- **API Key:** Free from [console.groq.com](https://console.groq.com)

System dependencies (auto-detected and installed during setup):

**X11:**
```bash
sudo apt install xclip xdotool
```

**Wayland:**
```bash
sudo apt install wl-clipboard wtype    # wlroots (Sway, Hyprland)
sudo apt install wl-clipboard ydotool  # GNOME, KDE, others
```

Undertone auto-detects your display server and uses the right tools.

---

## Text Cleanup

Undertone uses a two-stage cleanup pipeline:

1. **Regex filler removal** -- instantly strips "um", "uh", "like", "you know", etc.
2. **LLM grammar fix** -- sends text to Groq's llama-3.1-8b-instant for grammar, punctuation, and capitalization fixes (~160ms extra latency)

Toggle LLM cleanup on/off:
```
undertone> /config
# Select "LLM grammar fix on/off"
```

When LLM is off or fails, regex cleanup runs as fallback.

---

## Privacy Mode

By default, audio goes to Groq for fast transcription. Want to keep it local?

```
undertone> /config
# Select "Privacy mode" -> "local"
```

Local mode uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on your CPU. Slower (~3-5s) but your voice never leaves your machine.

---

## Troubleshooting

<details>
<summary><strong>No audio input detected</strong></summary>

- Check your mic is connected
- Run `arecord -l` to list audio devices
</details>

<details>
<summary><strong>Text not appearing</strong></summary>

- **X11:** `sudo apt install xclip xdotool`
- **Wayland (Sway/Hyprland):** `sudo apt install wl-clipboard wtype`
- **Wayland (GNOME/KDE):** `sudo apt install wl-clipboard ydotool`
- Check logs with `/logs` to see which tools were detected
</details>

<details>
<summary><strong>Service won't start</strong></summary>

- Check logs: run `undertone` then `/logs`
- Verify API key at console.groq.com
</details>

---

## Uninstall

```bash
# Stop service
undertone
# > /stop
# > /quit

# Remove package
pipx uninstall undertone

# Remove config (optional)
rm -rf ~/.config/undertone
rm ~/.config/systemd/user/undertone.service
systemctl --user daemon-reload
```

---

## Tech Stack

- **STT:** [Groq Whisper API](https://groq.com) / [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Cleanup:** Regex filler removal + LLM grammar fix (Groq llama-3.1-8b-instant)
- **Audio:** [sounddevice](https://python-sounddevice.readthedocs.io/)
- **Hotkeys:** [pynput](https://pynput.readthedocs.io/)
- **CLI:** [Rich](https://rich.readthedocs.io/)

---

## Contributing

PRs and issues welcome!

<a href="https://github.com/oneKn8/undertone/issues">Report Bug</a>
-
<a href="https://github.com/oneKn8/undertone/issues">Request Feature</a>

---

## License

MIT - do whatever you want with it.

---

<p align="center">
  <sub>Built with caffeine and voice commands</sub>
</p>
