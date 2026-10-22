"""Typed errors. Operational errors are never reported as scientific uncertainty (NEI)."""

from __future__ import annotations


class ClaimCheckerError(Exception):
    """Base error carrying the failing stage and an actionable setup hint."""

    error_type = "error"
    operational = True

    def __init__(self, message: str, *, stage: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.hint = hint

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "stage": self.stage,
            "hint": self.hint,
        }


class InputValidationError(ClaimCheckerError):
    error_type = "invalid_input"
    operational = False


class ConfigError(ClaimCheckerError):
    error_type = "config_error"


class DataNotReadyError(ClaimCheckerError):
    error_type = "data_not_ready"


class DataIntegrityError(ClaimCheckerError):
    error_type = "data_integrity"


class IndexNotReadyError(ClaimCheckerError):
    error_type = "index_not_ready"


class StaleIndexError(IndexNotReadyError):
    error_type = "stale_index"


class ModelUnavailableError(ClaimCheckerError):
    error_type = "model_unavailable"


class ModelTimeoutError(ClaimCheckerError):
    error_type = "model_timeout"


class ModelOutputError(ClaimCheckerError):
    error_type = "model_invalid_output"


class ContextOverflowError(ClaimCheckerError):
    """The prompt would not fit the model context; Ollama would otherwise truncate it silently."""

    error_type = "context_overflow"


class FrozenSplitError(ClaimCheckerError):
    error_type = "frozen_split"


class RunMismatchError(ClaimCheckerError):
    error_type = "run_mismatch"


OLLAMA_SETUP_HINT = (
    "Install Ollama (https://ollama.com/download), start it with `ollama serve`, "
    "then download the model with `ollama pull {model}`. Run `claim-checker doctor` to verify."
)
DATA_SETUP_HINT = "Run `claim-checker prepare-data` (needs internet once)."
INDEX_SETUP_HINT = "Run `claim-checker build-index` after `claim-checker prepare-data`."
