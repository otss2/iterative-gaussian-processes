from functools import partial
from data import Dataset
from typing import Callable
from structs import ModelParams, TrainState
from metrics import mll_exact
import time
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def exact_gradient(
    train_state: TrainState,
    model_params: ModelParams,
    ds: Dataset,
    kernel_fn: Callable,
    kernel_grad_fn: Callable = None,
    feature_params_fn: Callable = None,
    n_features: int = None,
    warm_start: bool = None,
    solver: Callable = None,
    rtol_y: float = None,
    rtol_z: float = None,
    max_epochs: int = None,
    batch_size: int = None,
):
    return train_state, jax.grad(mll_exact, argnums=2)(ds, kernel_fn, model_params), 0, 0.0, 0.0, 0.0


@partial(jax.jit, static_argnums=(5, 6))
def _trace_estimator(
    H_inv_y: Array,
    left: Array,
    right: Array,
    model_params: ModelParams,
    ds: Dataset,
    kernel_grad_fn: Callable,
    batch_size: int = 1,
):
    n_samples = left.shape[1]
    noise_scale_grad = model_params.noise_scale * (jnp.dot(H_inv_y, H_inv_y) - jnp.sum(left * right) / n_samples)

    n, d = ds.x.shape
    padding = (batch_size - (n % batch_size)) % batch_size

    x = jnp.concatenate([ds.x, jnp.zeros((padding, d))], axis=0)
    H_inv_y = jnp.concatenate([H_inv_y, jnp.zeros(padding)], axis=0)
    left = jnp.concatenate([left, jnp.zeros((padding, n_samples))], axis=0)

    def _f(A, idx):
        mean = jnp.einsum("i,ij...,j->...", H_inv_y[idx], A, H_inv_y[:n])
        trace = jnp.einsum("ik,ij...,jk->...", left[idx], A, right) / n_samples
        return 0.5 * (mean - trace)

    def _scan(kernel_params_grad, idx):
        dK_batch = kernel_grad_fn(x[idx], x[:n], model_params.kernel_params)
        kernel_params_grad_batch = jax.tree.map(partial(_f, idx=idx), dK_batch)
        kernel_params_grad = jax.tree.map(jnp.add, kernel_params_grad, kernel_params_grad_batch)
        return kernel_params_grad, None

    xs = jnp.reshape(jnp.arange(x.shape[0]), (-1, batch_size))
    kernel_params_grad, _ = jax.lax.scan(_scan, jax.tree.map(jnp.zeros_like, model_params.kernel_params), xs)

    return ModelParams(noise_scale=noise_scale_grad, kernel_params=kernel_params_grad)


def standard_gradient(
    train_state: TrainState,
    model_params: ModelParams,
    train_ds: Dataset,
    test_ds: Dataset,
    kernel_fn: Callable,
    kernel_grad_fn: Callable,
    feature_params_fn: Callable,
    n_features: int,
    warm_start: bool,
    solver: Callable,
    rtol_y: float,
    rtol_z: float,
    max_epochs: int,
    batch_size: int,
):
    v0, b, key = _standard_init(train_state, train_ds.x, train_ds.y, warm_start)

    solver_start = time.time()
    v, solver_iters, r_norm_y, r_norm_z = solver(
        v0,
        b,
        model_params,
        train_ds.x,
        kernel_fn,
        rtol_y,
        rtol_z,
        max_epochs,
        batch_size,
    )
    jax.block_until_ready(v)
    solver_end = time.time()
    solver_time = solver_end - solver_start

    model_params_grad = _trace_estimator(v[:, 0], v[:, 1:], b[:, 1:], model_params, train_ds, kernel_grad_fn)

    train_state = TrainState(key=key, v0=v, z=train_state.z, feature_params=None, w=None, eps=None, f0_train=None, f0_test=None)
    return train_state, model_params_grad, solver_iters, solver_time, r_norm_y, r_norm_z


def pathwise_gradient(
    train_state: TrainState,
    model_params: ModelParams,
    train_ds: Dataset,
    test_ds: Dataset,
    kernel_fn: Callable,
    kernel_grad_fn: Callable,
    feature_params_fn: Callable,
    feature_vec_prod_fn: Callable,
    n_features: int,
    warm_start: bool,
    solver: Callable,
    rtol_y: float,
    rtol_z: float,
    max_epochs: int,
    batch_size: int,
):
    v0, b, key, feature_params, w, eps, f0_train, f0_test = _pathwise_init(
        train_state,
        model_params,
        train_ds.x,
        test_ds.x,
        train_ds.y,
        feature_params_fn,
        feature_vec_prod_fn,
        n_features,
        warm_start,
        batch_size,
    )

    solver_start = time.time()
    v, solver_iters, r_norm_y, r_norm_z = solver(
        v0,
        b,
        model_params,
        train_ds.x,
        kernel_fn,
        rtol_y,
        rtol_z,
        max_epochs,
        batch_size,
    )
    jax.block_until_ready(v)
    solver_end = time.time()
    solver_time = solver_end - solver_start

    model_params_grad = _trace_estimator(v[:, 0], v[:, 1:], v[:, 1:], model_params, train_ds, kernel_grad_fn)

    train_state = TrainState(key=key, v0=v, z=None, feature_params=feature_params, w=w, eps=eps, f0_train=f0_train, f0_test=f0_test)
    return train_state, model_params_grad, solver_iters, solver_time, r_norm_y, r_norm_z


@partial(jax.jit, static_argnums=(3,))
def _standard_init(train_state: TrainState, x: Array, y: Array, warm_start: bool):
    if warm_start:
        v0 = train_state.v0
        key = train_state.key
        z = train_state.z
    else:
        v0 = jnp.zeros_like(train_state.v0)
        n_samples = train_state.v0.shape[1] - 1
        key, subkey = jr.split(train_state.key, 2)
        z = jr.normal(subkey, shape=(x.shape[0], n_samples))
    b = jnp.concatenate([y[:, None], z], axis=1)
    return v0, b, key


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9))
def _pathwise_init(
    train_state: TrainState,
    model_params: ModelParams,
    x_train: Array,
    x_test: Array,
    y: Array,
    feature_params_fn: Callable,
    feature_vec_prod_fn: Callable,
    n_features: int,
    warm_start: bool,
    batch_size: int,
):
    if warm_start:
        v0 = train_state.v0
        key = train_state.key
        feature_params = train_state.feature_params
        w = train_state.w
        eps = train_state.eps
    else:
        v0 = jnp.zeros_like(train_state.v0)
        n_samples = train_state.v0.shape[1] - 1
        key, feature_params_key, w_key, eps_key = jr.split(train_state.key, 4)
        feature_params = feature_params_fn(feature_params_key, x_train.shape[1], n_features)
        w = jr.normal(w_key, shape=(n_features, n_samples))
        n_tot = x_train.shape[0] + x_test.shape[0]
        eps = jr.normal(eps_key, shape=(n_tot, n_samples))
    f0_train, f0_test = feature_vec_prod_fn(x_train, x_test, eps, model_params, feature_params, w, batch_size)

    e = model_params.noise_scale * eps[: x_train.shape[0]]  # just augment to noisy samples for now
    z = f0_train + e
    b = jnp.concatenate([y[:, None], z], axis=1)
    return v0, b, key, feature_params, w, eps, f0_train, f0_test
