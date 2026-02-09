"""Runner module for systemd service."""

import logging
import sys

from undertone.config import get_api_key, load_config
from undertone.core import UndertoneEngine


def main():
    """Run the undertone engine."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()
    api_key = get_api_key()

    if not api_key:
        logging.error("No API key configured. Run 'undertone' to set up.")
        sys.exit(1)

    engine = UndertoneEngine(config, api_key=api_key)
    engine.run()


if __name__ == "__main__":
    main()
