import secrets
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class RunContext:
    run_id: str
    started_at: str


def create_run_context(run_id: str | None = None) -> RunContext:
    now = datetime.now(timezone.utc)
    if run_id:
        return RunContext(run_id=run_id, started_at=now.isoformat())
    token = secrets.token_hex(3)
    return RunContext(
        run_id=f"{now:%Y%m%dT%H%M%S.%fZ}-{token}",
        started_at=now.isoformat(),
    )
