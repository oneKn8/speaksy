"""Global hotkey listener for push-to-talk and toggle modes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeAlias

from pynput import keyboard

log = logging.getLogger("undertone")

KeyType: TypeAlias = keyboard.Key | keyboard.KeyCode


def _parse_key(key_str: str) -> KeyType:
    """Parse 'Key.ctrl_r' or 'Key.f8' to a pynput key object."""
    if key_str.startswith("Key."):
        attr = key_str[4:]
        return getattr(keyboard.Key, attr)
    return keyboard.KeyCode.from_char(key_str)


class HotkeyManager:
    """Global hotkey listener for push-to-talk and toggle modes."""

    def __init__(
        self,
        push_to_talk_key: str,
        toggle_key: str,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        self.ptt_key: KeyType = _parse_key(push_to_talk_key)
        self.toggle_key: KeyType = _parse_key(toggle_key)
        self.on_start = on_start
        self.on_stop = on_stop
        self._ptt_held = False
        self._toggle_active = False
        self._listener: keyboard.Listener | None = None

    def start(self) -> None:
        """Start the keyboard listener daemon."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        log.info(
            f"Hotkeys active: hold {self.ptt_key} (push-to-talk), press {self.toggle_key} (toggle)"
        )

    def _on_press(self, key: KeyType) -> None:
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

    def _on_release(self, key: KeyType) -> None:
        if key == self.ptt_key and self._ptt_held:
            self._ptt_held = False
            if not self._toggle_active:
                self.on_stop()

    def stop(self) -> None:
        """Stop the keyboard listener."""
        if self._listener:
            self._listener.stop()
