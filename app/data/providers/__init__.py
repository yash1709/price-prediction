from app.core.config import Settings
from app.data.providers.base import DataProvider, ProviderResult


def get_provider(settings: Settings) -> DataProvider:
    """Factory: add new providers here (e.g. an official NSE / licensed vendor API)."""
    name = settings.data_provider.lower()
    if name == "yfinance":
        from app.data.providers.yfinance_provider import YFinanceProvider
        return YFinanceProvider()
    if name == "csv":
        from app.data.providers.csv_provider import CSVProvider
        return CSVProvider(settings.csv_path)
    raise ValueError(f"Unknown DATA_PROVIDER '{settings.data_provider}' (use 'yfinance' or 'csv').")


__all__ = ["DataProvider", "ProviderResult", "get_provider"]
