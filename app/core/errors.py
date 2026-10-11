"""Domain exceptions. The API layer maps these to HTTP status codes."""
from __future__ import annotations


class AppError(Exception):
    code = "app_error"
    status_code = 500

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidSymbolError(AppError):
    code, status_code = "invalid_symbol", 422


class SymbolNotFoundError(AppError):
    code, status_code = "symbol_not_found", 404


class SymbolMismatchError(AppError):
    code, status_code = "symbol_mismatch", 502


class ProviderRateLimitError(AppError):
    code, status_code = "rate_limited", 429


class ProviderError(AppError):
    code, status_code = "provider_error", 502


class DataUnavailableError(AppError):
    code, status_code = "data_unavailable", 503


class InsufficientDataError(AppError):
    code, status_code = "insufficient_data", 422


class NoModelError(AppError):
    code, status_code = "no_model", 409


class ConflictError(AppError):
    code, status_code = "conflict", 409


class NotFoundError(AppError):
    code, status_code = "not_found", 404
