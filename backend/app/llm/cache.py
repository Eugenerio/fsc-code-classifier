"""Content-addressed disk cache for LLM outputs: identical input -> identical output, across runs and processes.

Gemini's `seed` is best effort, so sampling alone cannot make the pipeline repeatable. The key hashes everything that
shapes the answer (model, prompt, system prompt, schema, input bytes, seed, temperature); changing any of them is a miss.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

VERSION = "1"  # bump to invalidate every stored output when prompt or schema semantics change


def digest(*parts: str | bytes) -> str:
    """Length-prefixed, so ("ab", "c") and ("a", "bc") never collide."""
    h = hashlib.sha256(VERSION.encode())
    for part in parts:
        data = part.encode() if isinstance(part, str) else part
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def default_cache() -> LlmCache | None:
    """The app's shared output cache under `<cache_dir>/llm`, or None when LLM_CACHE=false."""
    from app.config import get_settings  # imported late: this module stays importable without the app's settings

    s = get_settings()
    return LlmCache(s.cache_dir / "llm") if s.llm_cache else None


class LlmCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str, ttl_s: float | None = None) -> str | None:
        try:
            entry = json.loads(self._path(key).read_text())
        except (OSError, ValueError):
            return None
        if ttl_s is not None and time.time() - entry.get("t", 0) > ttl_s:
            return None
        return entry.get("v")

    def put(self, key: str, value: str, *, replace: bool = False) -> str:
        """Store `value` and return what is stored. By default the first writer wins (atomic hard link), so concurrent
        processes converge on one answer; `replace=True` overwrites an expired entry. Write failures are non-fatal."""
        path = self._path(key)
        tmp = path.with_name(f".{uuid.uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"v": value, "t": time.time()}))
            if replace:
                os.replace(tmp, path)
            else:
                os.link(tmp, path)
        except FileExistsError:
            pass  # lost the race: another writer's answer is the canonical one
        except OSError:
            return value  # read-only or full disk: run uncached rather than fail the request
        finally:
            with contextlib.suppress(OSError):  # cleanup must not turn a cache problem into a failed request
                tmp.unlink(missing_ok=True)
        return self.get(key) or value
