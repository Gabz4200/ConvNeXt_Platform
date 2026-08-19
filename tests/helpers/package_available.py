import platform
from importlib.metadata import PackageNotFoundError, distribution

from lightning.pytorch import accelerators


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    :param package_name: The name of the package to be checked.

    :return: `True` if the package is available. `False` otherwise.
    """
    try:
        distribution(package_name)
    except PackageNotFoundError:
        return False
    return True


_TPU_AVAILABLE = (
    accelerators.TPUAccelerator.is_available()  # type: ignore[attr-defined]
    if hasattr(accelerators, "TPUAccelerator")
    else False
)

_IS_WINDOWS = platform.system() == "Windows"

_SH_AVAILABLE = not _IS_WINDOWS and _package_available("sh")

_DEEPSPEED_AVAILABLE = not _IS_WINDOWS and _package_available("deepspeed")
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _package_available("fairscale")

_WANDB_AVAILABLE = _package_available("wandb")
_NEPTUNE_AVAILABLE = _package_available("neptune")
_COMET_AVAILABLE = _package_available("comet_ml")
_MLFLOW_AVAILABLE = _package_available("mlflow")
