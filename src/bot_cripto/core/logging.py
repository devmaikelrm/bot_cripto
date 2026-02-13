"""Logging estructurado con structlog.

Produce JSON en producción y output legible en desarrollo.
Cada log entry incluye timestamp ISO, nivel, logger name y contexto.
"""

from __future__ import annotations

import logging
import sys

import structlog

from bot_cripto.core.config import LogFormat, get_settings


def setup_logging(level: str | None = None, fmt: LogFormat | None = None) -> None:
    """Configura structlog + stdlib logging.

    Args:
        level: Nivel de log (DEBUG, INFO, WARNING, ERROR). Default: config.
        fmt: Formato de salida (json o console). Default: config.
    """
    settings = get_settings()
    log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)
    log_format = fmt or settings.log_format

    # Procesadores comunes
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == LogFormat.JSON:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Silenciar loggers ruidosos
    for noisy in ["urllib3", "ccxt", "asyncio"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str, **initial_context: str) -> structlog.stdlib.BoundLogger:
    """Obtiene un logger con contexto inicial.

    Args:
        name: Nombre del logger (ej: "data.ingestion", "models.baseline").
        **initial_context: Contexto adicional (symbol, timeframe, job).

    Returns:
        Logger struct con contexto bindeado.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger
