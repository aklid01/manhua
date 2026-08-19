# Execution run context and timestamped run ID generation.

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone


# Immutable container for pipeline execution run ID and start timestamp
@dataclass(frozen=True)
class RunContext:
    run_id: str
    started_at: str


# Create execution context with existing run ID or new unique timestamped ID
def create_run_context(run_id: str | None = None) -> RunContext:
    now = datetime.now(timezone.utc)
    if run_id:
        return RunContext(run_id=run_id, started_at=now.isoformat())
    token = secrets.token_hex(3)
    return RunContext(
        run_id=f"{now:%Y%m%dT%H%M%S.%fZ}-{token}",
        started_at=now.isoformat(),
    )
