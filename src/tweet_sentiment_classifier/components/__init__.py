"""Components are imported lazily so that serving code (which only needs
DataTransformation) does not pull in the training stack (MLflow, seaborn, DL frameworks)."""
from importlib import import_module

_EXPORTS = {
    "DataIngestion": "data_ingestion",
    "DataValidation": "data_validation",
    "DataTransformation": "data_transformation",
    "ModelTrainer": "model_trainer",
    "BestModelSelector": "best_model_selector",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
