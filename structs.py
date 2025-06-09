from typing import NamedTuple, Optional
from chex import Array, PRNGKey


class Dataset(NamedTuple):
    x: Array
    y: Array
    N: int
    D: int

    mu_x: Optional[Array] = None
    sigma_x: Optional[Array] = None
    mu_y: Optional[Array] = None
    sigma_y: Optional[Array] = None


class KernelParams(NamedTuple):
    signal_scale: float
    length_scale: Array


class FeatureParams(NamedTuple):
    omega: Array
    phi: Array


class ModelParams(NamedTuple):
    noise_scale: float
    kernel_params: KernelParams


class TrainState(NamedTuple):
    key: PRNGKey
    v0: Array
    z: Array
    feature_params: FeatureParams
    w1: Array
    w2: Array
    eps: Array
    f0_train: Array  # GP prior draws at train points
    f0_test: Array  # GP prior draws at test points
