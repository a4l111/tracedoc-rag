"""Validated runtime model settings and Windows DPAPI-backed API keys."""

from __future__ import annotations

import base64
import ctypes
import json
import os
import sys
from ctypes import wintypes
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


ROOT = (Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[2])
SETTINGS_PATH = ROOT / "data" / "service" / "model_settings.json"
SECRET_PATH = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "TraceDoc-RAG" / "secrets.json"

DEFAULTS = {
    "deepseek": {"provider": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-chat",
                 "timeout": 30.0, "api_key_env": "DEEPSEEK_API_KEY"},
    "embedding": {"provider": "moark", "base_url": "https://api.moark.com/v1", "model": "Qwen3-Embedding-8B",
                  "dimensions": 4096, "timeout": 30.0, "api_key_env": "MOARK_API_KEY"},
    "reranker": {"provider": "moark", "base_url": "https://api.moark.com/v1", "model": "Qwen3-Reranker-8B",
                 "timeout": 30.0, "api_key_env": "MOARK_API_KEY"},
    "vision": {"provider": "moark", "base_url": "https://api.moark.com/v1",
               "model": "Qwen3-VL-30B-A3B-Instruct", "timeout": 60.0, "api_key_env": "MOARK_API_KEY"},
}


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _protect(value: str) -> str:
    if os.name != "nt":
        raise RuntimeError("Encrypted key storage currently requires Windows; use environment variables")
    raw = value.encode("utf-8")
    buffer = ctypes.create_string_buffer(raw)
    source = _Blob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output = _Blob()
    if not ctypes.windll.crypt32.CryptProtectData(ctypes.byref(source), "TraceDoc-RAG", None, None, None, 0,
                                                 ctypes.byref(output)):
        raise ctypes.WinError()
    try:
        return base64.b64encode(ctypes.string_at(output.pbData, output.cbData)).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)


def _unprotect(value: str) -> str:
    raw = base64.b64decode(value, validate=True)
    buffer = ctypes.create_string_buffer(raw)
    source = _Blob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output = _Blob()
    if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 0,
                                                   ctypes.byref(output)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output.pbData, output.cbData).decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate(role: str, value: dict) -> dict:
    if role not in DEFAULTS:
        raise ValueError("Unknown model role")
    expected = DEFAULTS[role]
    result = {**expected, **value}
    if result["provider"] != expected["provider"] or result["model"] != expected["model"]:
        raise ValueError(f"{role} must use {expected['provider']} / {expected['model']}")
    parsed = urlsplit(str(result["base_url"]))
    allowed_host = "api.deepseek.com" if role == "deepseek" else "api.moark.com"
    if parsed.scheme != "https" or parsed.hostname != allowed_host or parsed.username or parsed.password:
        raise ValueError(f"{role} base_url must use https://{allowed_host}")
    timeout = float(result["timeout"])
    if not 1 <= timeout <= 300:
        raise ValueError("timeout must be between 1 and 300 seconds")
    result["timeout"] = timeout
    return result


def load_settings() -> dict[str, dict]:
    saved = _read_json(SETTINGS_PATH)
    return {role: _validate(role, saved.get(role, {})) for role in DEFAULTS}


def save_role(role: str, value: dict) -> dict:
    current = _read_json(SETTINGS_PATH)
    validated = _validate(role, value)
    current[role] = {key: validated[key] for key in DEFAULTS[role] if key != "api_key_env"}
    _write_json(SETTINGS_PATH, current)
    return validated


def set_secret(role: str, api_key: str) -> None:
    if role not in DEFAULTS:
        raise ValueError("Unknown model role")
    api_key = api_key.strip()
    if not api_key or api_key in {"your_key_here", "replace_with_your_key"}:
        raise ValueError("API key is empty or a placeholder")
    secrets = _read_json(SECRET_PATH)
    encrypted = _protect(api_key)
    targets = ("embedding", "reranker", "vision") if DEFAULTS[role]["provider"] == "moark" else (role,)
    for target in targets:
        secrets[target] = encrypted
    _write_json(SECRET_PATH, secrets)


def delete_secret(role: str) -> None:
    secrets = _read_json(SECRET_PATH)
    targets = ("embedding", "reranker", "vision") if DEFAULTS[role]["provider"] == "moark" else (role,)
    changed = False
    for target in targets:
        if target in secrets:
            del secrets[target]
            changed = True
    if changed:
        _write_json(SECRET_PATH, secrets)


def secret_status(role: str) -> str:
    load_dotenv(ROOT / ".env", override=False)
    settings = load_settings()[role]
    if os.getenv(settings["api_key_env"], "").strip():
        return "environment"
    return "encrypted_store" if role in _read_json(SECRET_PATH) else "missing"


def get_api_key(role: str) -> str:
    load_dotenv(ROOT / ".env", override=False)
    settings = load_settings()[role]
    environment = os.getenv(settings["api_key_env"], "").strip()
    if environment:
        return environment
    encrypted = _read_json(SECRET_PATH).get(role)
    if not encrypted:
        raise ValueError(f"{settings['api_key_env']} is missing and no encrypted {role} key is saved")
    return _unprotect(encrypted)


def public_settings() -> dict[str, dict]:
    return {role: {**settings, "key_status": secret_status(role)}
            for role, settings in load_settings().items()}
