from functools import partial
from typing import Callable
from structs import Dataset, ModelParams, TrainState
from kernels import feature_vec_prod
import jax
import jax.numpy as jnp
from chex import Array


def _rmse(y_pred, y_gt):
    return jnp.sqrt(jnp.mean((y_pred - y_gt) ** 2))


def _llh(y_pred_mean, y_pred_std, y_gt):
    return jnp.mean(
        jax.scipy.stats.norm.logpdf(y_gt, loc=y_pred_mean, scale=y_pred_std)
    )


@partial(jax.jit, static_argnums=1)
def mll_exact(train_ds: Dataset, kernel_fn: Callable, model_params: ModelParams):
    K_train = kernel_fn(train_ds.x, train_ds.x, model_params.kernel_params)

    H = K_train + (model_params.noise_scale**2) * jnp.identity(train_ds.x.shape[0])
    H_cho_factor, lower = jax.scipy.linalg.cho_factor(H)
    H_inv_y = jax.scipy.linalg.cho_solve((H_cho_factor, lower), train_ds.y)

    data_fit_term = -0.5 * jnp.dot(train_ds.y, H_inv_y)
    log_det_term = -jnp.log(jnp.diag(H_cho_factor)).sum()
    const_term = -(train_ds.x.shape[0] / 2.0) * jnp.log(2.0 * jnp.pi)

    return data_fit_term + log_det_term + const_term


@partial(jax.jit, static_argnums=2)
def compute_metrics_exact(
    train_ds: Dataset, test_ds: Dataset, kernel_fn: Callable, model_params: ModelParams
):
    K_train = kernel_fn(train_ds.x, train_ds.x, model_params.kernel_params)
    K_test_train = kernel_fn(test_ds.x, train_ds.x, model_params.kernel_params)
    K_test = kernel_fn(test_ds.x, test_ds.x, model_params.kernel_params)

    H = K_train + (model_params.noise_scale**2) * jnp.identity(train_ds.x.shape[0])

    H_cho_factor, lower = jax.scipy.linalg.cho_factor(H)
    H_inv_y = jax.scipy.linalg.cho_solve((H_cho_factor, lower), train_ds.y)
    H_inv_K_train = jax.scipy.linalg.cho_solve((H_cho_factor, lower), K_train)
    H_inv_K_train_test = jax.scipy.linalg.cho_solve(
        (H_cho_factor, lower), K_test_train.T
    )

    data_fit_term = -0.5 * jnp.dot(train_ds.y, H_inv_y)
    log_det_term = -jnp.log(jnp.diag(H_cho_factor)).sum()
    const_term = -(train_ds.x.shape[0] / 2.0) * jnp.log(2.0 * jnp.pi)

    mll = data_fit_term + log_det_term + const_term

    y_pred_mean = K_train @ H_inv_y
    y_pred_var = K_train - K_train @ H_inv_K_train
    y_pred_std = jnp.sqrt(jnp.diag(y_pred_var) + (model_params.noise_scale**2))

    train_rmse = _rmse(y_pred_mean, train_ds.y)
    train_llh = _llh(y_pred_mean, y_pred_std, train_ds.y)

    y_pred_mean = K_test_train @ H_inv_y
    y_pred_var = K_test - K_test_train @ H_inv_K_train_test
    y_pred_std = jnp.sqrt(jnp.diag(y_pred_var) + (model_params.noise_scale**2))

    test_rmse = _rmse(y_pred_mean, test_ds.y)
    test_llh = _llh(y_pred_mean, y_pred_std, test_ds.y)

    return mll, train_rmse, train_llh, test_rmse, test_llh


@partial(jax.jit, static_argnums=2)
def compute_metrics_samples(
    train_ds: Dataset,
    test_ds: Dataset,
    kernel_fn: Callable,
    model_params: ModelParams,
    train_state: TrainState,
    batch_size: int = 1,
):
    def kvp(x: Array, v: Array):
        n, d = x.shape
        padding = (batch_size - (n % batch_size)) % batch_size
        x = jnp.concatenate([x, jnp.zeros((padding, d))], axis=0)

        def _f(_, idx):
            return _, kernel_fn(x[idx], train_ds.x, model_params.kernel_params) @ v

        xs = jnp.reshape(jnp.arange(x.shape[0]), (-1, batch_size))
        stack = jax.lax.scan(_f, jnp.zeros(()), xs)[1]
        return jnp.squeeze(jnp.reshape(stack, (x.shape[0], -1)))[:n]

    temp = kvp(train_ds.x, train_state.v0)
    y_pred_mean = temp[:, 0]
    correction = temp[:, 1:]

    f0 = feature_vec_prod(
        train_ds.x, model_params.kernel_params, train_state.feature_params,
        train_state.w, batch_size=batch_size)
    
    y_pred_std = jnp.sqrt(
        jnp.mean((f0 - correction) ** 2, axis=1) + (model_params.noise_scale**2)
    )

    train_rmse = _rmse(y_pred_mean, train_ds.y)
    train_llh = _llh(y_pred_mean, y_pred_std, train_ds.y)

    temp = kvp(test_ds.x, train_state.v0)
    y_pred_mean = temp[:, 0]
    correction = temp[:, 1:]

    f0 = feature_vec_prod(
        test_ds.x, model_params.kernel_params, train_state.feature_params,
        train_state.w, batch_size=batch_size)
    
    y_pred_std = jnp.sqrt(
        jnp.mean((f0 - correction) ** 2, axis=1) + (model_params.noise_scale**2)
    )

    test_rmse = _rmse(y_pred_mean, test_ds.y)
    test_llh = _llh(y_pred_mean, y_pred_std, test_ds.y)

    return train_rmse, train_llh, test_rmse, test_llh
