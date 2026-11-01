"""The package logger: one name, no handlers, and levels that mean something.

A compiler is asked to do things it cannot do, and it usually does them
slightly wrong rather than refusing outright: ``f32`` matmul drops to the scalar
kernel because the hardware has no ``f32`` tensor cores, ``codegen_attention``
trails MLX on prefill shapes so a short sequence takes the fallback, autotune
fails for every candidate on a paravirtual GPU, a platform is missing the
tensor-core probe and re-lowers. Every one of those is a *decision*, and every
one of them used to be invisible outside ``plan.selections`` - which only helps
if you already have the plan in hand.

So the decisions go to ``logging.getLogger("aotolith")`` and nothing else does:

* ``DEBUG`` - per-operation detail: which implementation each value took, which
  kernel variant was emitted, the storage slot a value was assigned.
* ``INFO`` - one line per session-level event: kernel count and ladder choice at
  the end of compilation, fallback share, artifact written and read, the number
  of plans a bucket runtime built.
* ``WARNING`` - a requested feature was downgraded: tensor cores unavailable,
  autotune produced no winner, a native kernel refused a shape the fallback
  accepted, an artifact was read with a device profile that does not match.
* ``ERROR`` - a plan was produced that is expected to fail, or a run failed.

A library must not configure logging for its host, so this module installs only
a ``NullHandler``. The one exception is :func:`basic_config`, which exists so
that the ``aoto`` command - and only the ``aoto`` command - can turn the record
on without the user writing logging setup first. Importing the package never
prints, never warns, and never reads an environment variable.
"""

from __future__ import annotations

import logging
import os
from typing import Final

#: The single name everything under this package logs to. A host that wants
#: only this package's records installs one handler on this logger.
LOGGER_NAME: Final = "aotolith"

#: Environment variable that pre-arms the logger, so a process that never calls
#: :func:`basic_config` - a test suite, a service - can still be made verbose
#: without code changes. Read lazily, never at import.
LEVEL_ENV: Final = "AOTOLITH_LOG"

#: Attribute set on the one handler this module installs, so it can recognise
#: its own and never stack a second one.
_OWN_HANDLER: Final = "_aotolith_owned"


def get_logger(suffix: str | None = None) -> logging.Logger:
    """Return ``aotolith`` or one of its children."""

    return logging.getLogger(LOGGER_NAME if not suffix else f"{LOGGER_NAME}.{suffix}")


def install_null_handler() -> None:
    """Attach the placeholder handler that keeps ``logging`` quiet.

    Called once from the package ``__init__``. Idempotent, because a reload of
    the module must not stack handlers.
    """

    logger = get_logger()
    if any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        return
    logger.addHandler(logging.NullHandler())


def level_from_environment() -> int | None:
    """Return the level named by :data:`LEVEL_ENV`, or ``None``.

    Accepts a level name (``DEBUG``) or a number (``10``); anything else is
    ignored rather than raising, because a typo in an environment variable
    should not stop a process from starting.
    """

    raw = os.environ.get(LEVEL_ENV)
    if not raw:
        return None
    named = logging.getLevelNamesMapping().get(raw.strip().upper())
    if named is not None:
        return named
    try:
        return int(raw)
    except ValueError:
        return None


def basic_config(level: int | str | None = None) -> None:
    """Give the package logger a stderr handler, once.

    Only the command line calls this. A library user who wants output installs
    their own handler on :data:`LOGGER_NAME` and never calls this.

    The handler is marked rather than tracked by a module flag, so calling this
    twice cannot stack handlers, and removing the handler (which a host is
    entitled to do) is not a state this function remembers wrongly.
    """

    resolved = level if level is not None else level_from_environment()
    if resolved is None:
        resolved = logging.WARNING
    logger = get_logger()
    handler = next(
        (item for item in logger.handlers if getattr(item, _OWN_HANDLER, False)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s: %(message)s")
        )
        setattr(handler, _OWN_HANDLER, True)
        logger.addHandler(handler)
    logger.setLevel(resolved)


__all__ = [
    "LEVEL_ENV",
    "LOGGER_NAME",
    "basic_config",
    "get_logger",
    "install_null_handler",
    "level_from_environment",
]
