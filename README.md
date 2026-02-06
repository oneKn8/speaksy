<p align="center">
  <img src="https://img.shields.io/badge/speaksy-voice%20typing-blueviolet?style=for-the-badge&logo=microphone" alt="speaksy">
</p>

<h1 align="center">speaksy</h1>

<p align="center">
  <strong>talk it. type it. ship it.</strong>
</p>

<p align="center">
  <a href="https://github.com/oneKn8/speaksy/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"></a>
  <a href="https://github.com/oneKn8/speaksy"><img src="https://img.shields.io/badge/platform-Linux-orange.svg" alt="Platform"></a>
  <a href="https://console.groq.com"><img src="https://img.shields.io/badge/powered%20by-Groq-ff6600.svg" alt="Groq"></a>
</p>

<p align="center">
  <em>Voice typing for Linux that actually works.<br>Hold a key, speak, release — your words appear wherever you're typing.</em>
</p>

---

## Demo

```
$ speaksy

  ╭────────────────────────────────────────╮
  │  SPEAKSY                               │
  │  talk it. type it. ship it.            │
  ╰────────────────────────────────────────╯

  Status: vibing
  Hotkeys: Right Ctrl (hold) | F8 (toggle)

speaksy> _
```

<!-- TODO: Add demo GIF here -->
<!-- ![Demo](assets/demo.gif) -->

---

## Quick Start

```bash
# Install
pipx install speaksy

# Run (interactive setup on first launch)
speaksy
```

That's it. 30 seconds to voice typing.

---

## Features

| | Feature | Description |
|---|---------|-------------|
| **Speed** | < 1 second latency | Groq's Whisper API is blazing fast |
| **Smart** | Filler removal | Auto-removes "um", "uh", "like", "you know" |
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
                   │ + filler removal│
                   └─────────────────┘
```

1. Press hotkey (Right Ctrl = hold, F8 = toggle)
2. Speak naturally
3. Release — text appears in < 1 second

Works everywhere: browser, terminal, IDE, Slack, Discord, anywhere you type.

---

## Commands

Run `speaksy` to open the interactive CLI:

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

- **OS:** Linux (X11 or XWayland)
- **Python:** 3.10+
- **API Key:** Free from [console.groq.com](https://console.groq.com)

System dependencies (auto-installed during setup):
```bash
sudo apt install xclip xdotool
```

---

## Privacy Mode

By default, audio goes to Groq for fast transcription. Want to keep it local?

```
speaksy> /config
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

- Install dependencies: `sudo apt install xclip xdotool`
- Some pure Wayland apps may not work with xdotool
</details>

<details>
<summary><strong>Service won't start</strong></summary>

- Check logs: run `speaksy` then `/logs`
- Verify API key at console.groq.com
</details>

---

## Uninstall

```bash
# Stop service
speaksy
# > /stop
# > /quit

# Remove package
pipx uninstall speaksy

# Remove config (optional)
rm -rf ~/.config/speaksy
rm ~/.config/systemd/user/speaksy.service
systemctl --user daemon-reload
```

---

## Tech Stack

- **STT:** [Groq Whisper API](https://groq.com) / [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Cleanup:** Regex-based filler word removal (no extra API calls)
- **Audio:** [sounddevice](https://python-sounddevice.readthedocs.io/)
- **Hotkeys:** [pynput](https://pynput.readthedocs.io/)
- **CLI:** [Rich](https://rich.readthedocs.io/)

---

## Contributing

PRs and issues welcome!

<a href="https://github.com/oneKn8/speaksy/issues">Report Bug</a>
·
<a href="https://github.com/oneKn8/speaksy/issues">Request Feature</a>

---

## License

MIT - do whatever you want with it.

---

<p align="center">
  <sub>Built with caffeine and voice commands</sub>
</p>
