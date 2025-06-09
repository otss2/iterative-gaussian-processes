from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class TrainConfig:
    seed: int
    dataset_name: str
    dataset_split: int
    n_subsample: int
    subsample_seed: int

    kernel_name: str
    n_features: int
    use_symmetric_features: bool

    pretrained_init: bool
    noise_scale_init: float
    signal_scale_init: float
    length_scale_init: float

    use_estimated_init: bool
    estimator_subsample_size: int
    relative_noise_scale: float

    noise_scale_min: float

    n_iterations: int
    learning_rate: float

    estimator_name: str
    warm_start: bool
    n_samples: int
    pathwise_init: bool

    solver_name: str
    rtol_y: float
    rtol_z: float
    max_epochs: int

    batch_size: int

    log_wandb: bool
    log_verbose: bool
    log_metrics_exact: bool
    log_metrics_samples: bool

    checkpoint_interval: int

    wandb_project: str
    wandb_entity: str
    wandb_api_key: str


cs = ConfigStore.instance()
cs.store(name="config_schema", node=TrainConfig)
