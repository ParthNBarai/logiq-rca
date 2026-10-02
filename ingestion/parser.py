"""Deterministic log line parser using compiled regex patterns.

Architecture: Factory pattern so additional parsers (JSON, CEF, syslog) can be
registered at runtime without touching call-sites.
"""

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedLogLine:
    """All fields extracted from a single raw log line.

    Optional fields are None when the pattern is absent from the message text.
    ``ingestion_id`` is derived on-demand so the dataclass stays lightweight.
    """

    timestamp: datetime
    tenant_id: str
    level: str
    logger_class: str
    message: str
    component: str
    is_error: bool
    raw_line: str
    http_status: Optional[int] = None
    latency_ms: Optional[int] = None
    db_cluster: Optional[str] = None
    circuit_breaker_state: Optional[str] = None
    exception_type: Optional[str] = None

    @property
    def ingestion_id(self) -> str:
        """SHA-256 of the raw line — used as idempotency key across re-ingestions."""
        return hashlib.sha256(self.raw_line.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseLogParser(ABC):
    """Contract every concrete parser must fulfil."""

    @abstractmethod
    def can_parse(self, sample_line: str) -> bool:
        """Return True if this parser can handle the given log format."""

    @abstractmethod
    def parse(self, line: str) -> Optional[ParsedLogLine]:
        """Parse a single raw line.

        Returns None for blank lines or lines that don't match the format,
        rather than raising, so the pipeline can skip gracefully.
        """


# ---------------------------------------------------------------------------
# Standard parser  (format already confirmed from log analysis)
# ---------------------------------------------------------------------------


class StandardLogParser(BaseLogParser):
    """Parser for the format:

        <ISO8601_ms_Z>  [TENANT-X]  LEVEL  [com.package.ClassName]  <message>

    All compiled patterns are class-level to avoid re-compilation per call.
    """

    # -- main line structure -------------------------------------------------
    _LINE: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)"
        r"\s+\[(?P<tenant_id>[^\]]+)\]"
        r"\s+(?P<level>DEBUG|INFO |WARN |ERROR|FATAL|INFO|WARN|ERROR)"
        r"\s+\[(?P<logger_class>[^\]]+)\]"
        r"\s+(?P<message>.+)$"
    )

    # -- sub-field patterns inside message -----------------------------------
    _HTTP_STATUS: ClassVar[re.Pattern[str]] = re.compile(r"Status[=:]\s*(\d{3})")
    _LATENCY: ClassVar[re.Pattern[str]] = re.compile(r"Latency=(\d+)ms")
    _DB_CLUSTER: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:cluster|Target:)\s*\[([^\]]+)\]"
    )
    _CB_STATE: ClassVar[re.Pattern[str]] = re.compile(
        r"State Changed:\s*(\w+\s*->\s*\w+)"
    )
    _EXCEPTION: ClassVar[re.Pattern[str]] = re.compile(
        r"(\w+(?:\.\w+)*Exception)"
    )

    _ERROR_LEVELS: ClassVar[frozenset[str]] = frozenset({"ERROR", "WARN", "FATAL"})

    # -----------------------------------------------------------------------

    def can_parse(self, sample_line: str) -> bool:
        """Return True when the line matches the standard log structure."""
        return bool(self._LINE.match(sample_line.strip()))

    def parse(self, line: str) -> Optional[ParsedLogLine]:
        """Parse one raw log line into a ``ParsedLogLine``.

        Returns None for blank lines and unrecognised formats.
        """
        stripped = line.strip()
        if not stripped:
            return None

        match = self._LINE.match(stripped)
        if not match:
            logger.debug("Line did not match standard pattern: %.80s", stripped)
            return None

        g = match.groupdict()
        level = g["level"].strip()
        logger_class = g["logger_class"]
        message = g["message"]

        # ISO-8601 with 'Z' → datetime with UTC tzinfo
        timestamp = datetime.fromisoformat(g["timestamp"].replace("Z", "+00:00"))

        return ParsedLogLine(
            timestamp=timestamp,
            tenant_id=g["tenant_id"],
            level=level,
            logger_class=logger_class,
            message=message,
            component=logger_class.split(".")[-1],
            is_error=level in self._ERROR_LEVELS,
            raw_line=stripped,
            http_status=self._extract_http_status(message),
            latency_ms=self._extract_latency(message),
            db_cluster=self._extract_db_cluster(message),
            circuit_breaker_state=self._extract_cb_state(message),
            exception_type=self._extract_exception(message),
        )

    # -- private extractors --------------------------------------------------

    def _extract_http_status(self, message: str) -> Optional[int]:
        m = self._HTTP_STATUS.search(message)
        return int(m.group(1)) if m else None

    def _extract_latency(self, message: str) -> Optional[int]:
        m = self._LATENCY.search(message)
        return int(m.group(1)) if m else None

    def _extract_db_cluster(self, message: str) -> Optional[str]:
        m = self._DB_CLUSTER.search(message)
        return m.group(1) if m else None

    def _extract_cb_state(self, message: str) -> Optional[str]:
        m = self._CB_STATE.search(message)
        return m.group(1) if m else None

    def _extract_exception(self, message: str) -> Optional[str]:
        m = self._EXCEPTION.search(message)
        return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class LogParserFactory:
    """Returns the first registered parser that claims to handle the format.

    Parsers are evaluated in registration order; the last ``register_parser``
    call gets the lowest priority. The built-in ``StandardLogParser`` is
    always registered as the default fallback.
    """

    _parsers: ClassVar[list[BaseLogParser]] = [StandardLogParser()]

    @classmethod
    def get_parser(cls, sample_line: str) -> BaseLogParser:
        """Return a parser capable of handling the given sample line.

        Raises ``ValueError`` when no registered parser matches.
        """
        for parser in cls._parsers:
            if parser.can_parse(sample_line.strip()):
                return parser
        raise ValueError(
            f"No registered parser handles this log format. "
            f"Sample (first 120 chars): {sample_line[:120]!r}"
        )

    @classmethod
    def register_parser(cls, parser: BaseLogParser) -> None:
        """Prepend a new parser so it takes precedence over existing ones."""
        cls._parsers.insert(0, parser)
