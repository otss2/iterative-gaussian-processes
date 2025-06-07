from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class TrainConfig:
    seed: int = 12345

    dataset_name: str = "housing"
    dataset_split: int = 0
    n_subsample: int = 0
    subsample_seed: int = 0

    kernel_name: str = "matern32"
    n_features: int = 2000
    use_symmetric_features: bool = True

    pretrained_init: bool = False
    noise_scale_init: float = 1.0
    signal_scale_init: float = 1.0
    length_scale_init: float = 1.0

    use_estimated_init: bool = False
    estimator_subsample_size: int = 10000
    relative_noise_scale: float = 0.2

    noise_scale_min: float = 0.0

    n_iterations: int = 100
    learning_rate: float = 1e-1

    estimator_name: str = "pathwise"
    warm_start: bool = True
    n_samples: int = 64
    pathwise_init: bool = False

    solver_name: str = "cg"
    rtol_y: float = 0.01
    rtol_z: float = 0.01
    max_epochs: int = 1000

    batch_size: int = 500

    log_wandb: bool = True
    log_verbose: bool = False
    log_metrics_exact: bool = False
    log_metrics_samples: bool = True

    checkpoint_interval: int = 0

    wandb_project: str = "symmetric-rff-experiments"
    wandb_entity: str = "otss2-university-of-cambridge"
    wandb_api_key: str = "bac5057ce3ab2f118779653c6f1a81ce21beb58a"


cs = ConfigStore.instance()
cs.store(name="config_schema", node=TrainConfig)
