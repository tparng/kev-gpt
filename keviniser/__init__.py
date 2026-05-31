"""Keviniser: telegraphic corpus preprocessor for the Kevin-on-Kria project."""

from .keviniser import (
    compression,
    keviniser,
    kevinise_stream,
    load_nlp,
)

__all__ = ["keviniser", "kevinise_stream", "compression", "load_nlp"]
