"""Post-parse normalisation and lightweight enrichment of log lines.

Keeps the parser focused on extraction; this layer enforces conventions
(canonical level names, whitespace trimming, tenant-id casing) without
touching the heavy regex logic.
"""

import logging
from copy import copy
from typing import ClassVar, Optional

from ingestion.parser import ParsedLogLine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


class LogNormalizer:
    """Applies deterministic normalisation rules to a ``ParsedLogLine``.

    All rules are stateless and idempotent — running the same record through
    ``normalize`` twice produces the same result.
    """

    _LEVEL_ALIASES: ClassVar[dict[str, str]] = {
        "WARNING": "WARN",
        "SEVERE": "ERROR",
        "CRITICAL": "ERROR",
        "TRACE": "DEBUG",
    }
    _CANONICAL_LEVELS: ClassVar[frozenset[str]] = frozenset(
        {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"}
    )
    _ERROR_LEVELS: ClassVar[frozenset[str]] = frozenset({"ERROR", "WARN", "FATAL"})

    def normalize(self, parsed: ParsedLogLine) -> ParsedLogLine:
        """Return a normalised copy of *parsed*.

        Never mutates the original so the pipeline can inspect the raw value
        alongside the normalised one if needed.
        """
        result = copy(parsed)

        result.level = self._normalize_level(parsed.level)
        result.is_error = result.level in self._ERROR_LEVELS
        result.message = parsed.message.strip()
        result.tenant_id = parsed.tenant_id.upper().strip()

        return result

    def normalize_batch(
        self, records: list[ParsedLogLine]
    ) -> list[ParsedLogLine]:
        """Normalise a list of parsed lines, dropping any that become invalid."""
        out: list[ParsedLogLine] = []
        for record in records:
            try:
                out.append(self.normalize(record))
            except Exception:
                logger.warning(
                    "Normalisation failed for record %s; skipping.",
                    record.raw_line[:80],
                    exc_info=True,
                )
        return out

    # -----------------------------------------------------------------------

    def _normalize_level(self, level: str) -> str:
        upper = level.upper().strip()
        canonical = self._LEVEL_ALIASES.get(upper, upper)
        if canonical not in self._CANONICAL_LEVELS:
            logger.debug("Unknown log level %r; defaulting to INFO.", level)
            return "INFO"
        return canonical
