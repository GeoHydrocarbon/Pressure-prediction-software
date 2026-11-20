"""Lightweight task state shared by training and I/O workers."""
import hashlib
import json
from pathlib import Path


class Cancelled(Exception):
    pass


def check_cancel(path):
    if path and Path(path).exists(): raise Cancelled("用户取消任务")


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def save_json(path, value):
    path = Path(path); temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def checked_sha256(path, cancel_file=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            check_cancel(cancel_file)
            chunk = stream.read(1024 * 1024)
            if not chunk: break
            digest.update(chunk)
    return digest.hexdigest()
