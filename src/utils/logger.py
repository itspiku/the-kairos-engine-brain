"""
The Kairos Engine - High-Contrast CLI Dashboard Logger

Production-grade logging with:
- Python logging module (replaces raw print)
- Console handler with colored, formatted output
- File handler with JSON-lines structured logging + rotation
- Audit trail logging to logs/kairos_audit.jsonl
"""

import os
import sys
import json
import time
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Ensure logs directory exists
# ---------------------------------------------------------------------------
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Custom formatter for rich console output
# ---------------------------------------------------------------------------
class _KairosConsoleFormatter(logging.Formatter):
    """Colored console formatter matching the original Kairos CLI style."""

    LEVEL_TAGS = {
        logging.DEBUG:    "[DEBUG]",
        logging.INFO:     "[INFO]",
        logging.WARNING:  "[WARN]",
        logging.ERROR:    "[ERROR]",
        logging.CRITICAL: "[CRIT]",
    }

    def format(self, record):
        tag = self.LEVEL_TAGS.get(record.levelno, "[LOG]")
        return f"   {tag} {record.getMessage()}"


class _KairosJSONFormatter(logging.Formatter):
    """Structured JSON formatter for file logging."""

    def format(self, record):
        entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            entry["data"] = record.extra_data
        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Configure root kairos logger once on module load
# ---------------------------------------------------------------------------
_kairos_logger = logging.getLogger("kairos")
_kairos_logger.setLevel(logging.DEBUG)
_kairos_logger.propagate = False

if not _kairos_logger.handlers:
    # Console handler
    _console = logging.StreamHandler(sys.stdout)
    _console.setLevel(logging.INFO)
    _console.setFormatter(_KairosConsoleFormatter())
    _kairos_logger.addHandler(_console)

    # File handler with rotation (5 MB, 3 backups)
    try:
        _file = logging.handlers.RotatingFileHandler(
            str(LOGS_DIR / "kairos_engine.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        _file.setLevel(logging.DEBUG)
        _file.setFormatter(_KairosJSONFormatter())
        _kairos_logger.addHandler(_file)
    except Exception:
        pass  # Don't crash if log file can't be created


class KairosLogger:
    """
    High-contrast CLI dashboard logger.

    Provides both backwards-compatible static methods (header, info, ok, warn,
    error, decision) and new audit trail logging.
    """

    _logger = _kairos_logger

    @staticmethod
    def header(title: str):
        """Print a prominent section header."""
        line = "=" * 64
        print(f"\n{line}")
        print(f"  ⚡ THE KAIROS ENGINE | {title.upper()}")
        print(line)
        KairosLogger._logger.info(f"=== {title.upper()} ===")

    @staticmethod
    def info(msg: str):
        """Log an informational message."""
        print(f"   [INFO] {msg}")
        KairosLogger._logger.info(msg)

    @staticmethod
    def ok(msg: str):
        """Log a success message."""
        print(f"   [OK] {msg}")
        KairosLogger._logger.info(f"OK: {msg}")

    @staticmethod
    def warn(msg: str):
        """Log a warning message."""
        print(f"   [WARN] {msg}")
        KairosLogger._logger.warning(msg)

    @staticmethod
    def error(msg: str):
        """Log an error message."""
        print(f"   [ERROR] {msg}")
        KairosLogger._logger.error(msg)

    @staticmethod
    def critical(msg: str):
        """Log a critical message."""
        print(f"   [CRITICAL] {msg}")
        KairosLogger._logger.critical(msg)

    @staticmethod
    def decision(decision: str, meets_sla: bool = True):
        """Log a flight decision with SLA status."""
        sla_tag = "[SLA PASS <2s]" if meets_sla else "[SLA WARN >2s]"
        print(f"\n   🎯 KAIROS DECISION: {decision}")
        print(f"   ⚡ LATENCY STATUS : {sla_tag}\n")
        KairosLogger._logger.info(f"DECISION: {decision} | {sla_tag}")

    @staticmethod
    def audit(event: str, data: Optional[Dict[str, Any]] = None):
        """
        Write a structured audit event to logs/kairos_audit.jsonl.
        Used for CAAN regulatory compliance and decision traceability.
        """
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "data": data or {},
        }
        try:
            audit_path = LOGS_DIR / "kairos_audit.jsonl"
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            KairosLogger._logger.error(f"Audit write failed: {exc}")

        KairosLogger._logger.info(f"AUDIT: {event}")

    @staticmethod
    def debug(msg: str):
        """Log a debug message (file only, not printed to console)."""
        KairosLogger._logger.debug(msg)
