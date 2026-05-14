# logger.py
#
# WHAT: Central logging configuration for the travel planner API.
#
# WHY DO WE NEED A SEPARATE LOGGER MODULE?
#   If you call logging.basicConfig() in every file, Python applies it only once
#   (the first call wins, all others are ignored). Centralizing it here ensures
#   ONE consistent format across all modules.
#
# HOW TO USE IN ANY FILE:
#   from logger import get_logger
#   logger = get_logger(__name__)
#   logger.info("Something happened")
#   logger.debug("Detailed value: %s", some_var)
#   logger.error("It broke", exc_info=True)   # exc_info=True prints full traceback
#
# WHERE DO YOU SEE THE LOGS?
#   LOCAL (2 terminals):
#     Terminal running uvicorn shows all logs directly in stdout.
#   DOCKER COMPOSE:
#     docker compose logs -f fastapi      ← live log tail for FastAPI container
#     docker compose logs -f streamlit    ← live log tail for Streamlit container
#     docker compose logs --tail=100      ← last 100 lines from all services
#   AZURE:
#     Portal → Container Apps → travel-api → Log stream
#     Or: az containerapp logs show --name travel-api --resource-group travel-planner-rg
#   LANGSMITH (for agent-level traces, not Python logs):
#     https://smith.langchain.com → your project → every LLM call, tool call is logged
#
# LOG LEVELS (in order of severity):
#   DEBUG   → granular detail (variable values, intermediate state) — dev only
#   INFO    → normal operation milestones (request received, agent routed, response sent)
#   WARNING → something unexpected but recoverable (empty result, retry)
#   ERROR   → something broke, needs attention (exception, DB failure)
#   CRITICAL→ system cannot continue (DB unreachable on startup, etc.)
#
# PRODUCTION RULE:
#   Set LOG_LEVEL=INFO in production (not DEBUG — too noisy, potential data leaks).
#   Set LOG_LEVEL=DEBUG only when actively debugging a specific issue.

import logging
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# LOG LEVEL FROM ENVIRONMENT
# WHY? You can change verbosity without touching code.
#   In .env:            LOG_LEVEL=DEBUG
#   In docker-compose:  LOG_LEVEL: ${LOG_LEVEL:-INFO}
#   In production:      LOG_LEVEL=WARNING  (less noise, lower cost if paid logging)
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTER
# WHY THIS FORMAT?
#   asctime     = when it happened  → timestamp for timeline reconstruction
#   name        = which module      → tells you WHERE the log came from
#   levelname   = severity          → INFO / DEBUG / ERROR at a glance
#   message     = what happened     → the actual info
#
# Example output:
#   2025-01-15 14:23:05,123 | orchestrator   | INFO  | Routing to researcher
#   2025-01-15 14:23:05,456 | researcher     | DEBUG | ReAct loop: 3 messages
#   2025-01-15 14:23:06,789 | main           | ERROR | Workflow failed: ...
# ─────────────────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(name)-15s | %(levelname)-5s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger() -> None:
    """
    Configure the root logger once at import time.
    All loggers created with get_logger() inherit this configuration.

    WHY StreamHandler(sys.stdout)?
      Docker captures stdout/stderr → your logs appear in `docker compose logs`.
      If you used a FileHandler instead, logs would be inside the container
      filesystem — invisible unless you exec into the container.
      stdout = the right choice for containerized apps.
    """
    root = logging.getLogger()

    # Avoid duplicate handlers if this module is imported multiple times
    if root.handlers:
        return

    root.setLevel(LOG_LEVEL)

    # ── Console handler — writes to stdout ──────────────────────────────────
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(LOG_LEVEL)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)

    # ── Silence noisy third-party loggers ───────────────────────────────────
    # These libraries log a LOT at DEBUG level — they'd flood your output.
    # We set them to WARNING so only their actual problems surface.
    for noisy_lib in ["httpx", "httpcore", "openai", "langchain", "urllib3"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


# Run once when this module is first imported
_configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Usage:
        from logger import get_logger
        logger = get_logger(__name__)

    WHY __name__?
        Python sets __name__ to the module's dotted path, e.g. "agents.orchestrator".
        This means your log output shows exactly which file the log came from.
        Much better than calling every logger "app" or "logger".
    """
    return logging.getLogger(name)
