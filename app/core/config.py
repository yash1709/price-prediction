"""Application settings, loaded from environment variables / `.env`."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    database_url: str = "sqlite:///data/nse.db"
    artifacts_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")
    data_provider: str = "yfinance"  # yfinance | csv
    csv_data_dir: Path = Path("data/csv")
    default_history_years: int = 8
    cache_ttl_hours: float = 6.0
    # While the newest bar is pending (no final price yet), re-check the provider this often.
    pending_retry_minutes: float = 15.0
    min_train_rows: int = 250
    walk_forward_folds: int = 8
    walk_forward_test_size: int = 21
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"
    universe_url: str = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    # A bar dated "today" (IST) is treated as incomplete until this IST hour (NSE closes 15:30).
    session_complete_hour_ist: float = 16.0

    def resolve_path(self, p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def artifacts_path(self) -> Path:
        return self.resolve_path(self.artifacts_dir)

    @property
    def data_path(self) -> Path:
        return self.resolve_path(self.data_dir)

    @property
    def csv_path(self) -> Path:
        return self.resolve_path(self.csv_data_dir)

    @property
    def resolved_database_url(self) -> str:
        """Make relative sqlite paths relative to the project root, not the CWD."""
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix) and not self.database_url.startswith("sqlite:////"):
            rel = self.database_url[len(prefix):]
            if rel and rel != ":memory:" and not Path(rel).is_absolute():
                return prefix + (PROJECT_ROOT / rel).as_posix()
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
