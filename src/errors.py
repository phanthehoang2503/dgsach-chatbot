class AppError(Exception):
    """Lỗi domain; message chỉ dùng để log, không trả cho client."""


class InvalidInputError(AppError, ValueError):
    """Input từ người dùng sai. Message an toàn để trả về client."""


class ConfigError(AppError, ValueError):
    """Server cấu hình sai (env, model, dimension)."""


class IndexNotReadyError(AppError, RuntimeError):
    """Bảng documents chưa có hoặc lệch fingerprint, cần chạy lại ingest."""


class UpstreamUnavailableError(AppError):
    """Ollama/Gemini/DB không phản hồi, timeout, quá tải."""
