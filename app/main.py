"""CLI-точка входа."""

from __future__ import annotations

import logging
import sys

from app.config import build_arg_parser, settings_from_sources
from app.pipeline import run


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args()
    settings = settings_from_sources(args)

    try:
        run(settings)
    except Exception:
        logging.getLogger(__name__).exception("Пайплайн завершился с ошибкой")
        sys.exit(1)


if __name__ == "__main__":
    main()
