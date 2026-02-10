"""System tray icon for Undertone."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PIL import Image, ImageDraw

log = logging.getLogger("undertone")

try:
    import pystray

    HAS_TRAY = True
except ImportError:
    pystray = None  # type: ignore[assignment]
    HAS_TRAY = False

TRAY_COLORS: dict[str, tuple[int, int, int, int]] = {
    "ready": (76, 175, 80, 255),
    "recording": (244, 67, 54, 255),
    "processing": (255, 193, 7, 255),
    "fallback": (255, 152, 0, 255),
}

TRAY_TITLES: dict[str, str] = {
    "ready": "Undertone - Ready",
    "recording": "Undertone - Recording...",
    "processing": "Undertone - Transcribing...",
    "fallback": "Undertone - Local Mode",
}


def _make_circle_icon(color: tuple[int, int, int, int], size: int = 64) -> Image.Image:
    """Create a solid circle icon for the system tray."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, size - 4, size - 4], fill=color)
    return img


class TrayManager:
    """System tray icon showing recording state."""

    def __init__(self, on_quit: Callable[[], None]) -> None:
        self.on_quit = on_quit
        self._icon: Any = None

    def start(self) -> None:
        """Start the system tray icon (blocking)."""
        if pystray is None:
            return
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

    def set_state(self, state: str) -> None:
        """Update the tray icon color and title."""
        if self._icon:
            self._icon.icon = _make_circle_icon(TRAY_COLORS.get(state, TRAY_COLORS["ready"]))
            self._icon.title = TRAY_TITLES.get(state, "Undertone")

    def _quit(self) -> None:
        if self._icon:
            self._icon.stop()
        self.on_quit()
