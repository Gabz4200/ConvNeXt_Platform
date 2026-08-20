"""Training entry point script for ConvNeXt Platform."""

import os

import hydra
import rootutils
from omegaconf import DictConfig

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, get_metric_value
from src.utils.trainer import run_train_task

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    extras(cfg)

    metric_dict, _ = run_train_task(cfg, task_name="training")

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()


from src.utils.trainer import run_train_task as train

__all__ = ["main", "train"]
