"""Fetch-on-demand for third-party Verilog-A sources that Voltax cannot bundle.

BSIM4's Verilog-A is licensed CC BY-NC 4.0, so it is not part of Voltax.
`vx.va.load("bsim4")` resolves it, in order, from

1. an explicit path (``vx.va.load("/path/to/bsim4.va")``),
2. the environment variable ``VOLTAX_BSIM4_VA`` (a path),
3. the cache, ``$XDG_CACHE_HOME/voltax/va/bsim4.va`` (default
   ``~/.cache/voltax/va/bsim4.va``),
4. a download from a pinned commit of github.com/dwarning/VA-Models, checked
   against a SHA-256 before use (the licence is shown once on download).

`fetch` performs step 4 explicitly and returns the cached path.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Remote:
    """A pinned, checksummed remote Verilog-A source."""

    url: str
    sha256: str
    env: str
    license: str


_VA_MODELS = "6099b4e12e6f66ff530f59b8d01fb909094be85a"  # dwarning/VA-Models main

REMOTES = {
    "bsim4": Remote(
        url=("https://raw.githubusercontent.com/dwarning/VA-Models/"
             f"{_VA_MODELS}/code/bsim4/vacode/bsim4.va"),
        sha256="7e6c0244249255680a0bb55fdb34a6bc961a0b4fa483005e37efb57f98e49cbe",
        env="VOLTAX_BSIM4_VA",
        license=(
            "BSIM4 4.8 Verilog-A (BSIM Group, UC Berkeley; Copyright 2001 Regents "
            "of the University of California; Verilog-A port from "
            "github.com/cogenda/VA-BSIM48, distributed via "
            "github.com/dwarning/VA-Models) is licensed CC BY-NC 4.0 "
            "(https://creativecommons.org/licenses/by-nc/4.0/): attribution "
            "required, NON-COMMERCIAL use only. It is not part of Voltax (MIT)."),
    ),
}
"""Sources `load`/`fetch` can download, by name."""

_NOTIFIED: set[str] = set()


def cache_dir() -> Path:
    """``$XDG_CACHE_HOME/voltax/va`` (default ``~/.cache/voltax/va``)."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "voltax" / "va"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(name: str = "bsim4", force: bool = False, timeout: float = 60.0) -> Path:
    """Download (if needed) the pinned source of model `name`; return its path.

    The file is verified against its SHA-256 and cached in `cache_dir`. Set
    `force` to download again even if a valid cached copy exists.

    Raises:
        KeyError: Unknown model name (see `REMOTES`).
        RuntimeError: Download failed or the checksum did not match; the
            message explains how to supply the file yourself.
    """
    remote = REMOTES[name]
    path = cache_dir() / f"{name}.va"
    if path.is_file() and not force:
        if _sha256(path.read_bytes()) == remote.sha256:
            return path
    help_ = (f"Download {remote.url} yourself and pass its path to "
             f"vx.va.load(path), or set the environment variable {remote.env} "
             "to it.")
    try:
        with urllib.request.urlopen(remote.url, timeout=timeout) as resp:
            data = resp.read()
    except Exception as e:  # noqa: BLE001 - any network failure
        raise RuntimeError(f"could not download the {name} Verilog-A source "
                           f"({e}). {help_}") from e
    if _sha256(data) != remote.sha256:
        raise RuntimeError(f"checksum mismatch for {remote.url} (expected "
                           f"{remote.sha256}). {help_}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
    os.replace(tmp.name, path)
    if name not in _NOTIFIED:
        _NOTIFIED.add(name)
        warnings.warn(f"downloaded {name} to {path}. {remote.license}",
                      stacklevel=2)
    return path


def resolve(name: str) -> Path | None:
    """Path of downloadable model `name` (env var, cache, then download), or
    None if `name` is not a downloadable model."""
    remote = REMOTES.get(name)
    if remote is None:
        return None
    env = os.environ.get(remote.env)
    if env:
        path = Path(env).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{remote.env}={env!r} is not a file")
        return path
    return fetch(name)
