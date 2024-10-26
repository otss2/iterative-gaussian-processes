from functools import partial
from typing import Callable
from kernels import KernelParams
from mll_optimiser import ModelParams
import jax
import jax.numpy as jnp
from chex import Array


norm_eps = 1e-10


def _A(
    v: Array,
    x: Array,
    kernel_fn: Callable,
    model_params: ModelParams,
    batch_size: int = 1,
):
    n, d = x.shape
    padding = (batch_size - (n % batch_size)) % batch_size
    x = jnp.concatenate([x, jnp.zeros((padding, d))], axis=0)

    def _f(_, idx):
        return _, kernel_fn(x[idx], x[:n], model_params.kernel_params) @ v

    xs = jnp.reshape(jnp.arange(x.shape[0]), (-1, batch_size))
    stack = jax.lax.scan(_f, jnp.zeros(()), xs)[1]
    kvp = jnp.reshape(stack, (-1, v.shape[1]))[:n]
    return kvp + (model_params.noise_scale**2) * v


def _pivoted_cholesky(
    x: Array, kernel_fn: Callable, kernel_params: KernelParams, rank: int = 100
):
    n = x.shape[0]
    L = jnp.zeros((rank, n))
    d = (kernel_params.signal_scale**2) * jnp.ones((n,))
    pi = jnp.arange(n)

    for m in range(rank):
        i = m + jnp.argmax(d[m:])
        temp = pi[m]
        pi = pi.at[m].set(pi[i])
        pi = pi.at[i].set(temp)

        L = L.at[m, pi[m]].set(jnp.sqrt(d[pi[m]]))

        a = kernel_fn(x[pi[m]], x[pi[m + 1 :]], kernel_params).squeeze()
        l = (a - jnp.sum(L[:m, pi[m], jnp.newaxis] * L[:m, pi[m + 1 :]], axis=0)) / L[
            m, pi[m]
        ]

        L = L.at[m, pi[m + 1 :]].set(l)
        d = d.at[pi[m + 1 :]].set(d[pi[m + 1 :]] - (l**2))
    return L


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def chol_solver(
    v0: Array,
    targets: Array,
    model_params: ModelParams,
    x: Array,
    kernel_fn: Callable,
    rtol_y: float = None,
    rtol_z: float = None,
    max_epochs: int = None,
    batch_size: int = None,
):
    K = kernel_fn(x, x, model_params.kernel_params)
    H = K + (model_params.noise_scale**2) * jnp.eye(x.shape[0])
    H_cho_factor, lower = jax.scipy.linalg.cho_factor(H)
    return jax.scipy.linalg.cho_solve((H_cho_factor, lower), targets), 0, 0.0, 0.0


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def cg_solver(
    v0: Array,
    targets: Array,
    model_params: ModelParams,
    x: Array,
    kernel_fn: Callable,
    rtol_y: float,
    rtol_z: float,
    max_epochs: int,
    batch_size: int,
):
    A = partial(
        _A, x=x, kernel_fn=kernel_fn, model_params=model_params, batch_size=batch_size
    )

    L = _pivoted_cholesky(x, kernel_fn, model_params.kernel_params)
    M = (model_params.noise_scale**-2) * L @ L.T + jnp.eye(L.shape[0])
    M_cho_factor, lower = jax.scipy.linalg.cho_factor(M)

    def _P(v):
        tmp = (
            (model_params.noise_scale**-2)
            * L.T
            @ jax.scipy.linalg.cho_solve((M_cho_factor, lower), L @ v)
        )
        return (model_params.noise_scale**-2) * (v - tmp)

    # P = lambda x: x
    P = _P
    
    targets_norm = jnp.linalg.norm(targets, axis=0, keepdims=True) + norm_eps
    targets = targets / targets_norm
    v0 = v0 / targets_norm

    def _cond(carry):
        _, r, _, _, _, k = carry
        r_norm = jnp.linalg.norm(r, axis=0)
        tol_y = r_norm[0] > rtol_y
        tol_z = jnp.mean(r_norm[1:]) > rtol_z
        return (tol_y | tol_z) & (k < max_epochs)

    def _body(carry):
        v, r, d, z, gamma, k = carry
        Ad = A(d)
        alpha = gamma / jnp.sum(d * Ad, axis=0)
        v = v + alpha[jnp.newaxis] * d
        r = r - alpha[jnp.newaxis] * Ad
        z = P(r)
        gamma_new = jnp.sum(r * z, axis=0)
        beta = gamma_new / gamma
        d = z + beta[jnp.newaxis] * d
        return v, r, d, z, gamma_new, k + 1

    r0 = targets - A(v0)
    z0 = P(r0)
    d0 = z0
    gamma0 = jnp.sum(r0 * z0, axis=0)

    carry_init = (v0, r0, d0, z0, gamma0, 0)
    v, r, _, _, _, k = jax.lax.while_loop(_cond, _body, carry_init)

    r_norm = jnp.linalg.norm(r, axis=0)
    r_norm_y = r_norm[0]
    r_norm_z = jnp.mean(r_norm[1:])

    v = v * targets_norm
    return v, k, r_norm_y, r_norm_z


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def ap_solver(
    v0: Array,
    targets: Array,
    model_params: ModelParams,
    x: Array,
    kernel_fn: Callable,
    rtol_y: float,
    rtol_z: float,
    max_epochs: int,
    batch_size: int,
):
    n = x.shape[0]
    n_blocks = (n - 1) // batch_size
    last_block_size = n - n_blocks * batch_size
    
    blocks = jnp.reshape(jnp.arange(n_blocks * batch_size), (n_blocks, batch_size))
    last_block = jnp.arange(n_blocks * batch_size, n)

    def _kernel_block(idx):
        return kernel_fn(x[idx], x[idx], model_params.kernel_params) + (
            model_params.noise_scale**2
        ) * jnp.eye(idx.shape[0])

    def _chol_mat(carry, i):
        return carry, jnp.linalg.cholesky(_kernel_block(blocks[i]))
    
    _, chol_mats = jax.lax.scan(_chol_mat, init=None, xs=jnp.arange(n_blocks))
    last_chol_mat = jnp.linalg.cholesky(_kernel_block(last_block))

    targets_norm = jnp.linalg.norm(targets, axis=0, keepdims=True) + norm_eps
    targets = targets / targets_norm
    v0 = v0 / targets_norm
    max_iters = (n / batch_size) * max_epochs

    def _block_r2(r):
        block_r2 = jnp.array([jnp.sum(r[block] ** 2, axis=0) for block in blocks])
        last_block_r2 = jnp.sum(r[last_block] ** 2, axis=0, keepdims=True)
        return jnp.concatenate([block_r2, last_block_r2], axis=0)

    def _cond(carry):
        _, _, block_r2, k = carry
        r_norm = jnp.sqrt(jnp.sum(block_r2, axis=0))
        tol_y = r_norm[0] > rtol_y
        tol_z = jnp.mean(r_norm[1:]) > rtol_z
        return (tol_y | tol_z) & (k < max_iters)

    def _f(idx, block_size, chol, v, r):
        K_batch = kernel_fn(x, x[idx], model_params.kernel_params)
        K_batch = K_batch.at[idx, jnp.arange(block_size)].set(
            K_batch[idx, jnp.arange(block_size)] + model_params.noise_scale**2
        )

        solve_batch = jax.scipy.linalg.cho_solve((chol, True), r[idx])
        return v.at[idx].set(v[idx] + solve_batch), r - K_batch @ solve_batch
    
    def _true_fun(chol_idx, v, r):
        idx = jax.lax.dynamic_slice(blocks, (chol_idx, 0), (1, batch_size)).squeeze()
        block_size = batch_size
        chol = jax.lax.dynamic_slice(
            chol_mats, (chol_idx, 0, 0), (1, batch_size, batch_size)
        ).squeeze()
        return _f(idx, block_size, chol, v, r)

    def _false_fun(_, v, r):
        return _f(last_block, last_block_size, last_chol_mat, v, r)

    def _body(carry):
        v, r, block_r2, k = carry
        chol_idx = jnp.argmax(jnp.sum(block_r2, axis=1))

        v, r = jax.lax.cond(chol_idx < n_blocks, _true_fun, _false_fun, chol_idx, v, r)

        return v, r, _block_r2(r), k + 1

    A = partial(
        _A, x=x, kernel_fn=kernel_fn, model_params=model_params, batch_size=batch_size
    )

    Av = A(v0)
    r0 = targets - Av
    carry_init = (v0, r0, _block_r2(r0), 0)

    v, _, block_r2, k = jax.lax.while_loop(_cond, _body, carry_init)

    r_norm = jnp.sqrt(jnp.sum(block_r2, axis=0))
    r_norm_y = r_norm[0]
    r_norm_z = jnp.mean(r_norm[1:])

    v = v * targets_norm
    return v, k, r_norm_y, r_norm_z


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9))
def sgd_solver(
    v0: Array,
    targets: Array,
    model_params: ModelParams,
    x: Array,
    kernel_fn: Callable,
    rtol_y: float,
    rtol_z: float,
    max_epochs: int,
    batch_size: int,
    random_seed: int = 0,
):
    momentum = 0.9
    lr = 1.0

    n = x.shape[0]
    key = jax.random.PRNGKey(random_seed)
    all_idx = jnp.arange(n)
    
    targets_norm = jnp.linalg.norm(targets, axis=0, keepdims=True) + norm_eps
    targets = targets / targets_norm
    v0 = v0 / targets_norm
    max_iters = (n / batch_size) * max_epochs

    @jax.jit
    def _cond(carry):
        _, _, r, k = carry
        r_norm = jnp.linalg.norm(r, axis=0)
        tol_y = r_norm[0] > rtol_y
        tol_z = jnp.mean(r_norm[1:]) > rtol_z
        return (tol_y | tol_z) & (k < max_iters)

    @jax.jit
    def _body(carry):
        v, velocity, r, k = carry
        loop_key = jax.random.fold_in(key, k)
        idx = jax.random.choice(loop_key, all_idx, shape=(batch_size,), replace=False)

        K_batch = kernel_fn(x[idx], x, model_params.kernel_params)
        g_batch = K_batch @ v - targets[idx] + (model_params.noise_scale**2) * v[idx]
        g = jnp.zeros_like(v)
        g = g.at[idx].set(g_batch) / batch_size

        velocity = momentum * velocity - lr * g
        v = v + velocity

        r = r.at[idx].set(g_batch)

        return v, velocity, r, k + 1

    r0 = targets
    velocity = jnp.zeros_like(v0)
    carry_init = (v0, velocity, r0, 0)

    v, _, r, k = jax.lax.while_loop(_cond, _body, carry_init)

    r_norm = jnp.linalg.norm(r, axis=0)
    r_norm_y = r_norm[0]
    r_norm_z = jnp.mean(r_norm[1:])

    v = v * targets_norm
    return v, k, r_norm_y, r_norm_z
