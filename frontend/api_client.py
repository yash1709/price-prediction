"""Thin HTTP client for the FastAPI backend (the only thing the UI knows about the backend)."""
from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import httpx


class ApiError(Exception):
    def __init__(self, message: str, code: str = "error", status: int | None = None):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _req(self, method: str, path: str, **kw) -> Any:
        if kw.get("params"):  # httpx would send None as an empty string -> 422 from the API
            kw["params"] = {k: v for k, v in kw["params"].items() if v is not None}
        try:
            r = httpx.request(method, self.base + path, timeout=self.timeout, **kw)
        except httpx.HTTPError as exc:
            raise ApiError(f"Cannot reach the API at {self.base} ({type(exc).__name__}). "
                           f"Start it with `python run.py api`.", "unreachable") from exc
        if r.status_code >= 400:
            try:
                err = r.json()["error"]
                raise ApiError(err["message"], err["code"], r.status_code)
            except (ValueError, KeyError):
                raise ApiError(f"HTTP {r.status_code}: {r.text[:300]}", "http_error",
                               r.status_code) from None
        return r.json()

    @staticmethod
    def _s(symbol: str) -> str:
        return quote(symbol, safe="")

    def health(self): return self._req("GET", "/health")

    def stocks(self, q: str | None = None, limit: int = 5000):
        return self._req("GET", "/stocks", params={"q": q, "limit": limit})

    def history(self, symbol, start=None, end=None, refresh=False):
        return self._req("GET", f"/stocks/{self._s(symbol)}/history",
                         params={"start": start, "end": end, "refresh": refresh})

    def train(self, symbol, body: dict):
        return self._req("POST", f"/stocks/{self._s(symbol)}/train", json=body)

    def task(self, task_id: str): return self._req("GET", f"/tasks/{task_id}")

    def models(self, symbol, horizon=None):
        return self._req("GET", f"/stocks/{self._s(symbol)}/models", params={"horizon": horizon})

    def backtest(self, symbol, model_id):
        return self._req("GET", f"/stocks/{self._s(symbol)}/models/{model_id}/backtest")

    def predict(self, symbol, body: dict):
        return self._req("POST", f"/stocks/{self._s(symbol)}/predict", json=body)

    def predictions(self, symbol, limit=500):
        return self._req("GET", f"/stocks/{self._s(symbol)}/predictions", params={"limit": limit})

    def performance(self, symbol): return self._req("GET", f"/stocks/{self._s(symbol)}/performance")

    def update_outcomes(self, symbol):
        return self._req("POST", f"/stocks/{self._s(symbol)}/outcomes/update")

    def wait_for_task(self, task_id: str, on_update=None, poll: float = 1.0,
                      timeout: float = 3600.0) -> dict:
        t0 = time.time()
        while True:
            t = self.task(task_id)
            if on_update:
                on_update(t)
            if t["status"] in ("succeeded", "failed"):
                return t
            if time.time() - t0 > timeout:
                raise ApiError("Timed out waiting for the task.", "timeout")
            time.sleep(poll)
