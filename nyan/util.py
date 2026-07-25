import os
import json
import random
from typing import TypeVar, Any
from collections.abc import Iterable
from datetime import datetime, timezone, timedelta, tzinfo, UTC
from dataclasses import dataclass, asdict, fields
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


# Base t.me URL of the channel this instance publishes to. Override per
# deployment via the PUBLISH_CHANNEL_URL env var.
# `or` rather than a getenv default: docker-compose passes unset variables
# through as empty strings, which a default would not replace.
PUBLISH_CHANNEL_URL = os.getenv("PUBLISH_CHANNEL_URL") or "https://t.me/UAliveNews"

# Timezone the channel's readers live in. A named zone rather than a fixed
# offset, so daylight saving time is handled instead of being an hour wrong for
# half the year.
DEFAULT_TIMEZONE = os.getenv("NYAN_TIMEZONE") or "Europe/Kyiv"

# Genitive case, which is what a Ukrainian date reads as ("25 липня"). The
# nominative names strftime would give ("Липень") are wrong in a date.
UK_MONTHS_GENITIVE = (
    "січня",
    "лютого",
    "березня",
    "квітня",
    "травня",
    "червня",
    "липня",
    "серпня",
    "вересня",
    "жовтня",
    "листопада",
    "грудня",
)


def get_timezone(name: str = DEFAULT_TIMEZONE) -> tzinfo:
    """The named timezone, falling back to UTC if tzdata is unavailable."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return UTC


def read_jsonl(file_path: str, sample_rate: float = 1.0) -> Iterable[dict[str, Any]]:
    assert os.path.exists(file_path)
    with open(file_path) as r:
        for line in r:
            if not line:
                continue
            if random.random() > sample_rate:
                continue
            yield json.loads(line)


def write_jsonl(file_path: str, records: list[dict[str, Any]]) -> None:
    with open(file_path, "w") as w:
        for record in records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


def get_current_ts() -> int:
    # `now(utc)`, not `now().replace(tzinfo=utc)`: the latter relabels local
    # wall-clock time as UTC, which is only correct when the host runs on UTC
    # and silently shifts every freshness calculation everywhere else.
    return int(datetime.now(UTC).timestamp())


def ts_to_dt(timestamp: int, tz: Any | None = None) -> datetime:
    """A timestamp as local time in `tz` (a zone name, tzinfo, or the default)."""
    if tz is None:
        tz = get_timezone()
    elif isinstance(tz, str):
        tz = get_timezone(tz)
    elif isinstance(tz, (int, float)):
        # Historic call style: a fixed UTC offset in hours.
        tz = timezone(timedelta(hours=tz))
    return datetime.fromtimestamp(timestamp, tz)


def format_dt_uk(dt: datetime, with_time: bool = True) -> str:
    """A date a Ukrainian reader parses at a glance: "25 липня, 14:30"."""
    date = f"{dt.day} {UK_MONTHS_GENITIVE[dt.month - 1]}"
    if not with_time:
        return date
    return "{}, {}".format(date, dt.strftime("%H:%M"))


T = TypeVar("T", bound="Serializable")


@dataclass
class Serializable:
    @classmethod
    def fromdict(cls: type[T], d: dict[str, Any]) -> T:
        keys = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in keys}
        return cls(**d)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def deserialize(cls: type[T], line: str) -> T:
        return cls.fromdict(json.loads(line))

    def serialize(self) -> str:
        return json.dumps(self.asdict(), ensure_ascii=False)


def set_random_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def gen_batch(records: list[Any], batch_size: int) -> Iterable[list[Any]]:
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start:batch_end]
        batch_start = batch_end
        yield batch


def normalize_url(url: str) -> str:
    """Normalize URL for consistent comparison and storage."""
    if not url:
        return ""
    normalized = url.strip()
    normalized = normalized.split("#", 1)[0]
    normalized = normalized.split("?", 1)[0]
    normalized = normalized.rstrip("/")
    return normalized.lower()


def normalize_channel_id(channel_id: str) -> str:
    """Normalize channel_id for consistent comparison and storage."""
    if not channel_id:
        return ""
    normalized = channel_id.lower().strip()

    if normalized.startswith("http://") or normalized.startswith("https://"):
        parsed = urlparse(normalized)
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts:
            if path_parts[0] == "s" and len(path_parts) > 1:
                normalized = path_parts[1]
            else:
                normalized = path_parts[0]

    return normalized.lstrip("@")
