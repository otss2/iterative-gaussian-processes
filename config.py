from typing import NamedTuple


class TrainConfig(NamedTuple):
    seed: int

    dataset_name: str
    dataset_split: int
    n_subsample: int
    subsample_seed: int

    kernel_name: str
    n_features: int

    pretrained_init: bool
    noise_scale_init: float
    signal_scale_init: float
    length_scale_init: float

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
