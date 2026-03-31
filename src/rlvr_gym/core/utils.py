from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def stable_hash_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def merge_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        merged[key] = value
    return merged


def as_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return as_primitive(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): as_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_primitive(item) for item in value]
    return value


def json_dumps(value: Any) -> str:
    return json.dumps(as_primitive(value), indent=2, sort_keys=True)
