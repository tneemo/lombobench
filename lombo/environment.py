"""Snapshot do ambiente de execução.

Registrado uma vez por combinação (fingerprint) — novo snapshot quando algo
muda, ex.: versão do Ollama. Cada request referencia o environment vigente.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

import httpx


def _ram_gb() -> float | None:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        return None


def _gpus() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "memory": parts[1], "driver": parts[2]})
        return gpus
    except Exception:
        return []


async def snapshot(ollama_url: str) -> dict[str, Any]:
    ollama_version = None
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ollama_url}/api/version", timeout=5)
            ollama_version = resp.json().get("version")
    except Exception:
        pass

    env = {
        "ollama_version": ollama_version,
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
        "ram_gb": _ram_gb(),
        "gpu_json": json.dumps(_gpus(), ensure_ascii=False),
    }
    env["fingerprint"] = hashlib.sha256(
        json.dumps(env, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    env["captured_at"] = datetime.now(timezone.utc).isoformat()
    return env
