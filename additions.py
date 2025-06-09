from chex import Array
from config import TrainConfig
import kernels
from structs import KernelParams, FeatureParams, ModelParams
from data import Dataset
import jax.numpy as jnp
from data import subsample
from kernels import _feature_fn_sin_cos
import jax
import jax.random as jr
import matplotlib.pyplot as plt


def estimate_init_params(train_ds: Dataset, cfg: TrainConfig):
    # signal scale is just standard deviation of the data
    # noise scale is just standard deviation of the data * relative noise scale
    # noise is std, not variance

    signal_scale_init = jnp.std(train_ds.y)
    noise_scale_init = signal_scale_init * cfg.relative_noise_scale

    # length scale is median of sq_dist (not including diagonal)
    # subsample the data
    subsampled_ds = subsample(train_ds, cfg.seed, cfg.estimator_subsample_size)

    length_scale_init = jnp.zeros(train_ds.D)
    for dim in range(train_ds.D):
        diff = subsampled_ds.x[:, dim : dim + 1] - subsampled_ds.x[:, dim : dim + 1].T
        sq_dist = diff**2

        # Get median of non-zero distances (exclude diagonal)
        mask = ~jnp.eye(subsampled_ds.N, dtype=bool)
        median_sq_dist = jnp.median(sq_dist[mask])
        if median_sq_dist < 1e-3:
            length_scale_init = length_scale_init.at[dim].set(1.0)
        else:
            length_scale_init = length_scale_init.at[dim].set(jnp.sqrt(median_sq_dist))

    return KernelParams(signal_scale=signal_scale_init, length_scale=length_scale_init), noise_scale_init


def feature_vec_prod_sym(
    x_train: Array,
    x_test: Array,
    eps: Array,  # unused, kept for signature compatibility
    model_params: ModelParams,
    feature_params: FeatureParams,
    vec: Array,  # this is your standard-normal u, shape (2D, num_samples)
    batch_size: int = 1,
):
    # Concatenate train and test points
    x = jnp.concatenate([x_train, x_test], axis=0)
    n_train = x_train.shape[0]

    num_features, num_samples = vec.shape
    num_features //= 2
    # kernel params
    kernel_params = model_params.kernel_params

    # build scaled frequencies
    omega = feature_params.omega / kernel_params.length_scale[:, None]

    # 1) form small Gram B = Phi^T Phi (2D x 2D), batched
    B = batched_build_B(x, omega, model_params.kernel_params.signal_scale, num_features, batch_size)

    # 2) eigendecompose B → V, eigvals so B = V S V^T
    eigvals, eigvecs = jnp.linalg.eigh(B)
    eigvals = jnp.maximum(eigvals, 1e-6)  # clamp small negatives
    S = jnp.sqrt(eigvals)  # singular values
    invS = 1.0 / S

    # 3) build the "M = V S^{-1}" transform (2D x 2D)
    M = eigvecs * invS[None, :]

    vec = eps  # need (N + N*, num_samples)

    # 4) form Z = Phi^T u (2D x num_samples)
    Z = batched_build_Z(
        x, omega,
        vec,
        model_params.kernel_params.signal_scale,
        num_features,
        batch_size,
        num_samples
    )

    # 5) projection coords in U-basis: alpha = M^T @ Z
    alpha = M.T @ Z                         # (2D x num_samples)

    # 6) prepare both M@alpha (for compatibility) and M@(alpha * S) (our sqrt(K) action)
    MA   = M @ alpha                        # (2D x num_samples), unused in final but needed by helper
    MA2  = M @ (alpha * S[:, None])         # scales by S

    # 7) run through your batched Phi @ MA and Phi @ MA2
    UA, UA2 = build_batched_UA(
        x, omega,
        MA, MA2,
        model_params.kernel_params.signal_scale,
        num_features,
        batch_size,
        num_samples
    )

    # 8) the sample from the prior is just the U-subspace part
    samples = UA2.squeeze()  # shape (N, num_samples)

    # this is train and test points, so we need to split
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    return train_samples, test_samples


@jax.jit
def feature_vec_prod_sym_noise(
    x: Array, eps: Array, model_params: ModelParams, feature_params: FeatureParams, vec: Array, batch_size: int = 1
):
    num_features, num_samples = vec.shape
    num_features //= 2
    # num_features is the number of frequencies
    # vec is what they call w (weights), shape (2*num_features, num_samples)
    # omega already generated as well, shape (x_dim, num_features)
    # eps is the fixed sample for the noise, we ignore it as noise built into this method
    kernel_params = model_params.kernel_params
    noise_scale = model_params.noise_scale  # std, not variance

    # omega aren't scaled by length, just raw
    omega = feature_params.omega / kernel_params.length_scale[:, None]

    # normal rffs - vec is (2*num_features, num_samples), we need (N, num_samples)
    vec = eps

    B = batched_build_B(x, omega, kernel_params.signal_scale, num_features, batch_size)

    # Eigendecompose B
    eigvals, eigvecs = jnp.linalg.eigh(B)
    eigvals = jnp.maximum(eigvals, 1e-6)
    S = jnp.sqrt(eigvals)
    invS = 1.0 / S
    M = eigvecs * invS[None, :]

    # Combine terms
    sqrt_noise = noise_scale
    sqrt_terms = jnp.sqrt(noise_scale**2 + S**2)

    # Z = Phi.T @ u
    Z = batched_build_Z(x, omega, vec, kernel_params.signal_scale, num_features, batch_size, num_samples)

    # alpha = U.T @ u, and U = Phi @ M, so alpha = M.T @ Phi.T @ u = M.T @ Z

    alpha = M.T @ Z  # (2D, num_samples)

    # We need U @ alpha and U @ (alpha * sqrt_terms)
    # U @ alpha = Phi @ M @ alpha
    MA = M @ alpha  # (2D, num_samples)
    # now need to run batched Phi @ MA
    # U @ (alpha * sqrt_terms) = Phi @ M @ (alpha * sqrt_terms)
    MA2 = M @ (alpha * sqrt_terms[:, None])  # (2D, num_samples)
    # now need to run batched Phi @ MA2 as well, can be done in the same scan

    UA, UA2 = build_batched_UA(x, omega, MA, MA2, kernel_params.signal_scale, num_features, batch_size, num_samples)

    r = vec - UA

    f1 = sqrt_noise * r  # in the orthogonal complement
    f2 = UA2  # in the U-subspace

    samples = f1 + f2  # (N, num_samples)

    # noise is built into the samples for symmetric features

    return samples.squeeze()


def batched_build_B(
    X, omega, signal_scale, num_features, batch_size
):  # (N, d)  # (2*num_features, d)  — your RFF frequencies  # scalar  # D
    N, d = X.shape
    D2 = 2 * num_features

    # Add padding to make X evenly divisible by batch_size
    padding = (batch_size - (N % batch_size)) % batch_size
    X_padded = jnp.concatenate([X, jnp.zeros((padding, d))], axis=0)

    def single_batch_B(B, X_batch):
        # X_batch: (b, d)
        # 1) WX = X_batch @ w.T           → (b, 2D)
        WX = X_batch @ omega

        # 2) form Phi_batch = [cos, sin]  → (b, 2D)
        #    *and* scale exactly as your one-shot transform does:
        Phi = jnp.concatenate([jnp.cos(WX), jnp.sin(WX)], axis=1)
        Phi = Phi * signal_scale / jnp.sqrt(num_features)

        # 3) accumulate B += Phi^T @ Phi
        return B + (Phi.T @ Phi), None

    # initialize
    B0 = jnp.zeros((D2, D2))

    # Reshape padded data and run single scan
    X_batches = jnp.reshape(X_padded, (-1, batch_size, d))
    B_final, _ = jax.lax.scan(single_batch_B, B0, X_batches)

    return B_final


def batched_build_Z(X, omega, vec, signal_scale, num_features, batch_size, num_samples):
    N, d = X.shape
    D2 = 2 * num_features

    # Add padding to make X evenly divisible by batch_size
    padding = (batch_size - (N % batch_size)) % batch_size
    X_padded = jnp.concatenate([X, jnp.zeros((padding, d))], axis=0)
    u_padded = jnp.concatenate([vec, jnp.zeros((padding, num_samples))], axis=0)

    def single_batch_Z(Z, Xu_batch):
        X_batch, u_batch = Xu_batch
        WX = X_batch @ omega

        Phi = jnp.concatenate([jnp.cos(WX), jnp.sin(WX)], axis=1)
        Phi = Phi * signal_scale / jnp.sqrt(num_features)

        return Z + (Phi.T @ u_batch), None

    Z0 = jnp.zeros((D2, num_samples))

    # Reshape padded data and run single scan
    X_batches = jnp.reshape(X_padded, (-1, batch_size, d))
    u_batches = jnp.reshape(u_padded, (-1, batch_size, num_samples))
    Z, _ = jax.lax.scan(single_batch_Z, Z0, (X_batches, u_batches))

    return Z


def build_batched_UA(X, omega, MA, MA2, signal_scale, num_features, batch_size, num_samples):
    N, dim = X.shape
    D2 = 2 * num_features

    padding = (batch_size - (N % batch_size)) % batch_size
    X_padded = jnp.concatenate([X, jnp.zeros((padding, dim))], axis=0)

    # Phi is (N, 2D)
    # MA, MA2 are (2D, num_samples)

    # So Phi @ MA is (N, num_samples)
    # when we batch it, we compute Phi_batch @ MA, which is (batch_size, num_samples)
    # then concat together to get (N, num_samples)

    def single_batch_UA(carry, X_batch):
        WX = X_batch @ omega
        Phi = jnp.concatenate([jnp.cos(WX), jnp.sin(WX)], axis=1)
        Phi = Phi * signal_scale / jnp.sqrt(num_features)

        UA_both = Phi @ MA_concat

        return carry, UA_both

    X_batches = jnp.reshape(X_padded, (-1, batch_size, dim))
    MA_concat = jnp.concatenate([MA, MA2], axis=1)  # (2D, 2*num_samples), this will still do it independently for MA and MA2
    _, UA_both = jax.lax.scan(single_batch_UA, None, X_batches)

    # UA both is now (num_batches, batch_size, 2*num_samples)
    # need to squeeze out the batch_size dimension, and trim off the padding
    UA_both = UA_both.reshape(-1, 2 * num_samples)
    UA_both = UA_both[:N, :]

    UA = UA_both[:, :num_samples]
    UA2 = UA_both[:, num_samples:]

    return UA, UA2
