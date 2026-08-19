# Runtime interaction policy and TTY detection.

import sys
from dataclasses import dataclass
from enum import Enum


# Interaction mode classifications for runtime execution
class InteractionMode(str, Enum):
    INTERACTIVE = "interactive"
    NON_INTERACTIVE = "non_interactive"
    MCP = "mcp"
    GUI = "gui"


# Encapsulates interaction mode and batch execution status
@dataclass(frozen=True)
class RuntimePolicy:
    mode: InteractionMode
    batch: bool = False

    # Check whether the session is interactive and connected to a TTY
    @property
    def can_prompt(self) -> bool:
        return (
            self.mode == InteractionMode.INTERACTIVE
            and not self.batch
            and sys.stdin.isatty()
        )


# Infer default runtime policy from environment and batch flag
def default_runtime_policy(*, batch: bool = False) -> RuntimePolicy:
    if batch:
        return RuntimePolicy(mode=InteractionMode.NON_INTERACTIVE, batch=True)
    if sys.stdin.isatty():
        return RuntimePolicy(mode=InteractionMode.INTERACTIVE, batch=False)
    return RuntimePolicy(mode=InteractionMode.NON_INTERACTIVE, batch=False)


# Check if both stdin and stdout are interactive TTYs
def can_prompt_now(*, batch: bool = False) -> bool:
    if batch:
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (ValueError, OSError):
        return False
