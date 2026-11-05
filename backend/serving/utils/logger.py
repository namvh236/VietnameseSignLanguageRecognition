"""Shared logging setup. Never use print() anywhere in the backend."""

import logging

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_configured = False

# Third-party libs log noisy emoji/info-level output -- keep only WARNING and above.
_NOISY_PREFIXES = ("Vieneu",)


class _ThirdPartyNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.levelno < logging.WARNING and record.name.startswith(_NOISY_PREFIXES))


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(level=level, format=_FORMAT)
    noise_filter = _ThirdPartyNoiseFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(noise_filter)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
