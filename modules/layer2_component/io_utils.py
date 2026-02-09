from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator, TextIO


def open_text(path: str | Path, mode: str) -> TextIO:
    """
    统一打开普通文本或 .gz 文本文件。
    - mode 仅支持文本模式：'r'/'w'/'a' 之一
    """
    p = Path(path)
    if "b" in mode:
        raise ValueError("open_text 仅支持文本模式（不要传 'b'）")
    if p.suffix == ".gz":
        return gzip.open(p, mode + "t", encoding="utf-8")  # type: ignore[return-value]
    return p.open(mode, encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl_line(f: TextIO, obj: dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

