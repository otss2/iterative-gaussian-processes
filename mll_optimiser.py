from jax import config as jax_config

from functools import partial
from typing import Callable
import additions
from config import TrainConfig
from data import Dataset
from additions import estimate_init_params
import kernels
from structs import KernelParams, ModelParams, TrainState
from metrics import compute_metrics_exact, compute_metrics_samples
from jax.nn import softplus, sigmoid
from optax import OptState, adam, apply_updates
import time
import jax
import jax.numpy as jnp
import jax.random as jr
import wandb
from chex import Array


def fit(
    train_ds: Dataset,
    test_ds: Dataset,
    kernel_fn: Callable,
    kernel_grad_fn: Callable,
    feature_params_fn: Callable,
    gradient_estimator: Callable,
    solver: Callable,
    cfg: TrainConfig,
    checkpoint: dict = None,
    transform_fn: Callable = softplus,
    transform_grad_fn: Callable = sigmoid,
    transform_inv_fn: Callable = lambda x: jnp.log(jnp.exp(x) - 1.0),
):
    if cfg.estimator_name not in ["exact", "standard", "pathwise"]:
        raise ValueError(f"Unknown estimator name {cfg.estimator_name}")

    if cfg.use_symmetric_features:
        feature_vec_prod_fn = additions.feature_vec_prod_sym
    else:
        feature_vec_prod_fn = kernels.feature_vec_prod_vanilla

    gradient_estimator = partial(
        gradient_estimator,
        train_ds=train_ds,
        test_ds=test_ds,
        kernel_fn=kernel_fn,
        kernel_grad_fn=kernel_grad_fn,
        feature_params_fn=feature_params_fn,
        feature_vec_prod_fn=feature_vec_prod_fn,
        n_features=cfg.n_features,
        warm_start=cfg.warm_start,
        solver=solver,
        rtol_y=cfg.rtol_y,
        rtol_z=cfg.rtol_z,
        max_epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
    )

    float_cast = jnp.float64 if jax_config.read("jax_enable_x64") else jnp.float32
    print("Using float64" if jax_config.read("jax_enable_x64") else "Using float32")

    noise_scale_u_min = transform_inv_fn(cfg.noise_scale_min) if cfg.noise_scale_min > 0.0 else -jnp.inf
    optimiser = adam(learning_rate=-cfg.learning_rate)

    if checkpoint is None:
        if cfg.pretrained_init:
            pretrained = jnp.load("./download/pretrained.npy", allow_pickle=True).item()
            model_params_init = pretrained[cfg.dataset_name][cfg.dataset_split]
        elif cfg.use_estimated_init:
            kernel_params_init, noise_scale_init = estimate_init_params(train_ds, cfg)
            model_params_init = ModelParams(noise_scale=noise_scale_init, kernel_params=kernel_params_init)
        else:
            kernel_params_init = KernelParams(
                signal_scale=cfg.signal_scale_init,
                length_scale=jnp.array(train_ds.D * [cfg.length_scale_init]),
            )
            model_params_init = ModelParams(noise_scale=cfg.noise_scale_init, kernel_params=kernel_params_init)

        model_params_init = jax.tree.map(float_cast, model_params_init)
        print("Initial model_params:")
        print(model_params_init)
        model_params_u = jax.tree.map(transform_inv_fn, model_params_init)
        opt_state = optimiser.init(model_params_u)

        if cfg.estimator_name == "exact":
            train_state = None
        else:
            key = jr.PRNGKey(cfg.seed)
            v0 = jnp.zeros((train_ds.N, cfg.n_samples + 1))
            if cfg.warm_start:
                if cfg.estimator_name == "standard":
                    key, z_key = jr.split(key)
                    z = jr.normal(z_key, shape=(train_ds.N, cfg.n_samples))
                    train_state = TrainState(key=key, v0=v0, z=z, feature_params=None, w1=None, w2=None, eps=None, f0_train=None, f0_test=None)
                elif cfg.estimator_name == "pathwise":
                    if cfg.pathwise_init:
                        train_state = jnp.load("./download/pathwise_init.npy", allow_pickle=True).item()[cfg.dataset_name][
                            cfg.dataset_split
                        ]
                    else:
                        key, feature_params_key, w1_key, w2_key, eps_key = jr.split(key, 5)
                        feature_params = feature_params_fn(feature_params_key, train_ds.D, cfg.n_features)
                        w1 = jr.normal(w1_key, shape=(cfg.n_features, cfg.n_samples))
                        w2 = jr.normal(w2_key, shape=(train_ds.N + test_ds.N, cfg.n_samples))
                        eps = jr.normal(eps_key, shape=(train_ds.N + test_ds.N, cfg.n_samples))
                        train_state = TrainState(key=key, v0=v0, z=None, feature_params=feature_params, w1=w1, w2=w2, eps=eps, f0_train=None, f0_test=None)
            else:
                train_state = TrainState(key=key, v0=v0, z=None, feature_params=None, w1=None, w2=None, eps=None, f0_train=None, f0_test=None)
        train_state = jax.tree.map(lambda x: float_cast(x) if isinstance(x, Array) and jnp.issubdtype(x, jnp.floating) else x, train_state)

        train_time = 0.0
        solver_time = 0.0
        eval_time = 0.0
        cumulative_solver_iters = 0
        i_init = 0
    else:
        print("Initial model_params:")
        print(checkpoint["model_params"])
        model_params_u = checkpoint["model_params_u"]
        opt_state = checkpoint["opt_state"]
        train_state = checkpoint["train_state"]
        train_time = checkpoint["train_time"]
        solver_time = checkpoint["solver_time"]
        eval_time = checkpoint["eval_time"]
        cumulative_solver_iters = checkpoint["cumulative_solver_iters"]
        i_init = checkpoint["step"]

    @jax.jit
    def _transform(model_params_u: ModelParams):
        return jax.tree.map(transform_fn, model_params_u)

    @jax.jit
    def _update(model_params_u: ModelParams, mll_grad: ModelParams, opt_state: OptState):
        transform_grad = jax.tree.map(transform_grad_fn, model_params_u)
        grad = jax.tree.map(jnp.multiply, mll_grad, transform_grad)

        updates, opt_state = optimiser.update(grad, opt_state)
        model_params_u = apply_updates(model_params_u, updates)

        noise_scale_u_clipped = jnp.clip(model_params_u.noise_scale, a_min=noise_scale_u_min, a_max=None)
        model_params_u = ModelParams(
            noise_scale=noise_scale_u_clipped,
            kernel_params=model_params_u.kernel_params,
        )
        return model_params_u, opt_state

    def update(train_state: TrainState, model_params_u: ModelParams, opt_state: OptState):
        model_params = _transform(model_params_u)

        train_state, mll_grad, solver_iters, solver_time, r_norm_y, r_norm_z = gradient_estimator(
            train_state=train_state, model_params=model_params
        )

        model_params_u, opt_state = _update(model_params_u, mll_grad, opt_state)

        return train_state, model_params_u, opt_state, solver_iters, solver_time, r_norm_y, r_norm_z

    def log(i, model_params_u, train_time, solver_time, solver_iters, cumulative_solver_iters, r_norm_y, r_norm_z, commit=True):
        model_params = _transform(model_params_u)
        if cfg.log_metrics_exact:
            mll, train_rmse, train_llh, test_rmse, test_llh = compute_metrics_exact(train_ds, test_ds, kernel_fn, model_params)

        if cfg.log_wandb:
            model_params_dict = {
                **{
                    "params/noise_scale": model_params.noise_scale,
                    "params/signal_scale": model_params.kernel_params.signal_scale,
                },
                **{f"params/length_scale_{i}": l for i, l in enumerate(model_params.kernel_params.length_scale)},
            }
            info = {"i": i, **model_params_dict, "train_time": train_time}

            if cfg.log_metrics_exact:
                info.update(
                    {
                        "mll": mll,
                        "train_rmse": train_rmse,
                        "train_llh": train_llh,
                        "test_rmse": test_rmse,
                        "test_llh": test_llh,
                    }
                )

            if cfg.estimator_name != "exact":
                info.update({"solver_time": solver_time})
                if cfg.solver_name != "cholesky":
                    info.update(
                        {
                            "solver_iters": solver_iters,
                            "cumulative_solver_iters": cumulative_solver_iters,
                            "r_norm_y": r_norm_y,
                            "r_norm_z": r_norm_z,
                        }
                    )

            wandb.log(info, commit=commit)

        if cfg.log_verbose:
            info_str = f"i: {i}, "

            if cfg.log_metrics_exact:
                info_str += f"mll: {mll:.4f}, train_rmse: {train_rmse:.4f}, train_llh: {train_llh:.4f}, test_rmse: {test_rmse:.4f}, test_llh: {test_llh:.4f}, "

            info_str += f"train_time: {train_time:.2f}"

            if cfg.estimator_name != "exact":
                info_str += f", solver_time: {solver_time:.2f}"
                if cfg.solver_name != "cholesky":
                    info_str += f", solver_iters: {solver_iters}, cumulative_solver_iters: {cumulative_solver_iters}, r_norm_y: {r_norm_y:.4f}, r_norm_z: {r_norm_z:.4f}"

            print(info_str, end="\n" if commit else ", ")

    running_pred = (cfg.estimator_name == "pathwise") and cfg.log_metrics_samples
    if (i_init == 0) and (cfg.log_wandb or cfg.log_verbose):
        log(0, model_params_u, 0.0, 0.0, 0, 0, 0.0, 0.0, commit=(not running_pred))

    for i in range(i_init, cfg.n_iterations + int(running_pred)):
        update_start = time.time()
        (train_state, model_params_u_next, opt_state, solver_iters, solver_time_i, r_norm_y, r_norm_z) = update(
            train_state, model_params_u, opt_state
        )
        jax.block_until_ready(model_params_u)
        update_end = time.time()

        solver_time += solver_time_i
        train_time += update_end - update_start
        cumulative_solver_iters += solver_iters

        if cfg.log_wandb or cfg.log_verbose:
            if running_pred:
                eval_start = time.time()
                (
                    train_rmse_samples,
                    train_llh_samples,
                    test_rmse_samples,
                    test_llh_samples,
                ) = compute_metrics_samples(
                    train_ds,
                    test_ds,
                    kernel_fn,
                    feature_vec_prod_fn,
                    _transform(model_params_u),
                    train_state,
                )
                eval_end = time.time()
                eval_time += eval_end - eval_start

                if cfg.log_wandb:
                    wandb.log(
                        {
                            "eval_time": eval_time,
                            "train_rmse_samples": train_rmse_samples,
                            "train_llh_samples": train_llh_samples,
                            "test_rmse_samples": test_rmse_samples,
                            "test_llh_samples": test_llh_samples,
                        },
                        commit=True,
                    )

                if cfg.log_verbose:
                    print(
                        f"eval_time: {eval_time:.4f}, "
                        f"train_rmse_samples: {train_rmse_samples:.4f}, "
                        f"train_llh_samples: {train_llh_samples:.4f}, "
                        f"test_rmse_samples: {test_rmse_samples:.4f}, "
                        f"test_llh_samples: {test_llh_samples:.4f}"
                    )
            if (not running_pred) or (i < cfg.n_iterations):
                log(
                    i + 1,
                    model_params_u_next,
                    train_time,
                    solver_time,
                    solver_iters,
                    cumulative_solver_iters,
                    r_norm_y,
                    r_norm_z,
                    commit=(not running_pred),
                )

        model_params_u = model_params_u_next

        if ((i + 1) - i_init) == cfg.checkpoint_interval:
            checkpoint = {
                "model_params_u": model_params_u,
                "model_params": _transform(model_params_u),
                "opt_state": opt_state,
                "train_state": train_state,
                "train_time": train_time,
                "solver_time": solver_time,
                "eval_time": eval_time,
                "cumulative_solver_iters": cumulative_solver_iters,
                "step": i + 1,
            }
            return checkpoint

    checkpoint = {
        "model_params_u": model_params_u,
        "model_params": _transform(model_params_u),
        "opt_state": opt_state,
        "train_state": train_state,
        "train_time": train_time,
        "solver_time": solver_time,
        "eval_time": eval_time,
        "cumulative_solver_iters": cumulative_solver_iters,
        "step": cfg.n_iterations + 1,
    }
    return checkpoint
