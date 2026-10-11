"""Run the FastAPI backend inside the Streamlit process (for single-process hosting such as
Streamlit Community Cloud). Same app, same HTTP API: the dashboard code path is unchanged."""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("x")
        probe.unlink()
        return True
    except OSError:
        return False


@st.cache_resource(show_spinner="Starting embedded API…")
def ensure_started() -> str:
    """Start (once per server process) a uvicorn thread serving the API; return its base URL."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if not (_writable(ROOT / "data") and _writable(ROOT / "artifacts")):
        tmp = Path(tempfile.gettempdir()) / "nse_app"  # read-only checkout: use temp storage
        os.environ.setdefault("DATABASE_URL", f"sqlite:///{(tmp / 'nse.db').as_posix()}")
        os.environ.setdefault("ARTIFACTS_DIR", str(tmp / "artifacts"))

    import uvicorn
    from app.api.main import create_app
    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app(get_settings())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, name="embedded-api", daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return url
        except httpx.HTTPError:
            time.sleep(0.2)
    raise RuntimeError("Embedded API failed to start within 20 s.")
