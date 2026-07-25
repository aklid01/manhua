"""Runtime interaction policy.

Determines whether the current execution context can prompt the user.
Validation and backend code must never call input() directly - they
consult RuntimePolicy.can_prompt first.
"""

import sys
from dataclasses import dataclass
from enum import Enum


class InteractionMode(str, Enum):
    INTERACTIVE = "interactive"
    NON_INTERACTIVE = "non_interactive"
    MCP = "mcp"
    GUI = "gui"


@dataclass(frozen=True)
class RuntimePolicy:
    mode: InteractionMode
    batch: bool = False

    @property
    def can_prompt(self) -> bool:
        """True only when running interactively in a real terminal, not batch/MCP/GUI."""
        return (
            self.mode == InteractionMode.INTERACTIVE
            and not self.batch
            and sys.stdin.isatty()
        )


def default_runtime_policy(*, batch: bool = False) -> RuntimePolicy:
    """Infer the runtime mode from the environment."""
    if batch:
        return RuntimePolicy(mode=InteractionMode.NON_INTERACTIVE, batch=True)
    if sys.stdin.isatty():
        return RuntimePolicy(mode=InteractionMode.INTERACTIVE, batch=False)
    return RuntimePolicy(mode=InteractionMode.NON_INTERACTIVE, batch=False)


def can_prompt_now(*, batch: bool = False) -> bool:
    """True only when a real interactive terminal is available."""
    if batch:
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (ValueError, OSError):
        return False

