"""Domain errors. `message` is user-facing (Spanish), `code` is machine-readable (English)."""

from __future__ import annotations


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}
