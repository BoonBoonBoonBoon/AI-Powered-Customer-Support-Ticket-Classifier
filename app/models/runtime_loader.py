import json
import os
import pathlib
import shutil
import urllib.parse
from typing import Dict, Any


REGISTRY_DIR = pathlib.Path("models/registry")
RUNTIME_CACHE = pathlib.Path(os.getenv("RUNTIME_CACHE_DIR", "runtime_cache"))


def _read_json(p: pathlib.Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_from_file_uri(uri: str, dest: pathlib.Path) -> pathlib.Path:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Only file:// URIs supported in local mode, got: {uri}")
    src_path = pathlib.Path(parsed.path.lstrip("/")) if os.name == "nt" else pathlib.Path(parsed.path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    return dest


def load_registry_pointer(env: str = "production") -> Dict[str, Any]:
    reg_file = REGISTRY_DIR / f"{env}.json"
    if not reg_file.exists():
        raise FileNotFoundError(f"Registry pointer missing: {reg_file}")
    return _read_json(reg_file)


def fetch_manifest(manifest_uri: str) -> Dict[str, Any]:
    RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
    local_manifest = RUNTIME_CACHE / "manifest.json"
    _copy_from_file_uri(manifest_uri, local_manifest)
    return _read_json(local_manifest)


def fetch_artifacts(artifacts: Dict[str, str]) -> Dict[str, pathlib.Path]:
    local_paths: Dict[str, pathlib.Path] = {}
    for k, uri in artifacts.items():
        if not uri:
            continue
        filename = os.path.basename(urllib.parse.urlparse(uri).path)
        dest = RUNTIME_CACHE / filename
        _copy_from_file_uri(uri, dest)
        local_paths[k] = dest
    return local_paths


def ensure_model_ready(env: str = "production") -> Dict[str, Any]:
    pointer = load_registry_pointer(env)
    manifest = fetch_manifest(pointer["manifest_uri"])
    # Minimal validation: required keys
    for key in ("model_id", "framework", "format", "artifacts", "metrics"):
        if key not in manifest:
            raise ValueError(f"Manifest missing required field: {key}")
    # Download artifacts locally (file:// for now)
    local_artifacts = fetch_artifacts({
        k: v for k, v in manifest.get("artifacts", {}).items() if v
    })
    return {"pointer": pointer, "manifest": manifest, "artifacts": local_artifacts}


__all__ = ["ensure_model_ready", "load_registry_pointer"]
