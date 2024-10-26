from typing import Optional
from structs import Dataset
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from uci_datasets import Dataset as uci_dataset
from uci_datasets import all_datasets


def _z_score(data: Array, mu: Optional[Array] = None, sigma: Optional[Array] = None):
    if (mu is not None) and (sigma is not None):
        return (data - mu) / sigma
    else:
        mu = jnp.mean(data, axis=0, keepdims=True)
        sigma = jnp.std(data, axis=0, keepdims=True)
        return (data - mu) / sigma, mu, sigma


def _matern_toy_data(
    n1: int = 200,
    n2: int = 200,
    d: int = 1,
    seed: int = 4,
    noise_scale: float = 0.15,
    signal_scale: float = 1.0,
    length_scale: float = 0.5,
):
    from kernels import KernelParams, matern32_kernel_fn

    kernel_params = KernelParams(
        signal_scale=signal_scale, length_scale=jnp.array(d * [length_scale])
    )

    key_x1, key_x2, key_shuffle, key_y = jr.split(jr.PRNGKey(seed), 4)

    x1 = jr.uniform(key_x1, shape=(n1, d), minval=-2, maxval=-1)
    x2 = jr.uniform(key_x2, shape=(n2, d), minval=0.5, maxval=2.5)

    N = n1 + n2
    x = jnp.concatenate([x1, x2], axis=0)
    x = jr.permutation(key_shuffle, x, axis=0, independent=True)

    K = matern32_kernel_fn(x, x, kernel_params)
    cov = K + (noise_scale**2) * jnp.eye(N)
    y = jr.multivariate_normal(key_y, mean=jnp.zeros((N,)), cov=cov)

    n_train = int(0.9 * N)
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    train_ds = Dataset(x_train, y_train, n_train, d)
    test_ds = Dataset(x_test, y_test, N - n_train, d)

    return train_ds, test_ds


def _uci_dataset(dataset_name: str, split: int = 0):
    dataset = uci_dataset(dataset_name)
    x_train, y_train, x_test, y_test = dataset.get_split(split)
    N, D = x_train.shape
    N_test, _ = x_test.shape

    train_ds = Dataset(jnp.array(x_train), jnp.array(y_train).squeeze(), N, D)
    test_ds = Dataset(jnp.array(x_test), jnp.array(y_test).squeeze(), N_test, D)

    return train_ds, test_ds


def _normalise_dataset(train_ds: Dataset, test_ds: Dataset):
    train_x, mu_x, sigma_x = _z_score(train_ds.x)
    train_y, mu_y, sigma_y = _z_score(train_ds.y)
    test_x = _z_score(test_ds.x, mu=mu_x, sigma=sigma_x)
    test_y = _z_score(test_ds.y, mu=mu_y, sigma=sigma_y)

    train_ds = Dataset(
        train_x, train_y, train_ds.N, train_ds.D, mu_x, sigma_x, mu_y, sigma_y
    )
    test_ds = Dataset(
        test_x, test_y, test_ds.N, test_ds.D, mu_x, sigma_x, mu_y, sigma_y
    )
    return train_ds, test_ds


def get_dataset(dataset_name, normalise: bool = True, **kwargs):
    if dataset_name == "matern_toy":
        train_ds, test_ds = _matern_toy_data(**kwargs)
    elif dataset_name in all_datasets:
        train_ds, test_ds = _uci_dataset(dataset_name, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if normalise:
        train_ds, test_ds = _normalise_dataset(train_ds, test_ds)

    return train_ds, test_ds


def subsample(ds: Dataset, subsample_seed: int, n_subsample: int):
    # return full dataset if it is smaller than subsample size
    if ds.N <= n_subsample:
        print("Dataset is smaller than n_subsample, returning full dataset.")
        return ds

    # choose random data point
    i = jr.randint(jr.PRNGKey(subsample_seed), (), minval=0, maxval=ds.N)

    # compute Euclidean distances to each data point
    # (could be replaced by any other distance measure if desired)
    dist = jnp.sum((ds.x[i, None] - ds.x[None, :]) ** 2, axis=-1).squeeze()
    idx = jnp.argsort(dist)[:n_subsample]

    # create subsampled dataset
    ds_subsample = Dataset(
        x=ds.x[idx],
        y=ds.y[idx],
        N=n_subsample,
        D=ds.D,
        mu_x=ds.mu_x,
        sigma_x=ds.sigma_x,
        mu_y=ds.mu_y,
        sigma_y=ds.sigma_y,
    )

    return ds_subsample
