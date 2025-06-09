from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
from structs import KernelParams, FeatureParams, ModelParams
from additions import feature_vec_prod_sym
from kernels import feature_vec_prod_vanilla, _matern_omega_fn, matern32_kernel_fn
import matplotlib.pyplot as plt
from jax.scipy.stats import multivariate_normal
from jax.scipy import linalg

# Set random seed for reproducibility
key = jr.PRNGKey(42)

# Generate some test data
n_points = 1000  # reduced number of points to avoid memory issues
x_dim = 1
x = jnp.linspace(-2, 2, n_points).reshape(-1, 1)

shuffle_idx = jr.permutation(key, jnp.arange(n_points))
x = x[shuffle_idx]
reverse_shuffle_idx = jnp.argsort(shuffle_idx)

test_split = 0.1
n_train = int(n_points * (1 - test_split))
n_test = n_points - n_train

x_train, x_test = x[:n_train], x[n_train:]

# Test parameters
n_freq = 1000
signal_scale = 1.4
noise_scale = 0.5  # increased noise for stability
length_scales = jnp.logspace(-0.5, -0.0, 8)
length_scales = [0.2, 1.0]

# Parameters for log-likelihood test
n_samples_ll = 500  # reduced number of samples for testing
test_length_scale = 0.2  # fixed length scale for likelihood test

# Create figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Comparison of Feature Vector Products")

n_samples = 1

# Generate random vectors and noise for visualization
vec_key, noise_key, omega_key = jr.split(key, 3)
vec = jr.normal(vec_key, shape=(2 * n_freq, n_samples))
eps = jr.normal(noise_key, shape=(n_points, n_samples))

omega = _matern_omega_fn(omega_key, x_dim, n_freq, df=3)
feature_params = FeatureParams(omega=omega, phi=None)

# First part: Visualization of samples
for i, length_scale in enumerate(length_scales):
    # Create kernel parameters
    kernel_params = KernelParams(signal_scale=signal_scale, length_scale=jnp.array([length_scale]))
    model_params = ModelParams(noise_scale=noise_scale, kernel_params=kernel_params)

    # Compute feature vector products
    samples_vanilla_train, samples_vanilla_test = feature_vec_prod_vanilla(x_train, x_test, eps, model_params, feature_params, vec)
    samples_sym_train, samples_sym_test = feature_vec_prod_sym(x_train, x_test, eps, model_params, feature_params, vec)


    # Plot results
    label = f"l={length_scale}"
    ax1.scatter(x_train, samples_vanilla_train, alpha=0.7, label=label, s=1, color="red")
    ax2.scatter(x_train, samples_sym_train, alpha=0.7, label=label, s=1, color="red")

    # plot test points
    ax1.scatter(x_test, samples_vanilla_test, alpha=0.7, label=label, s=1, color="blue")
    ax2.scatter(x_test, samples_sym_test, alpha=0.7, label=label, s=1, color="blue")

# Customize plots
ax1.set_title("Vanilla Feature Vector Product")
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.legend()
ax1.grid(True)

ax2.set_title("Symmetric Feature Vector Product")
ax2.set_xlabel("x")
ax2.set_ylabel("f(x)")
ax2.legend()
ax2.grid(True)

plt.show()
quit()

# Second part: Log-likelihood test
print("\nComputing log-likelihoods...")

# Generate new random vectors and noise for likelihood test
key, vec_key, noise_key, omega_key = jr.split(key, 4)
vec = jr.normal(vec_key, shape=(2 * n_freq, n_samples_ll))
eps = jr.normal(noise_key, shape=(n_points, n_samples_ll))

# Create kernel parameters for test
kernel_params = KernelParams(signal_scale=signal_scale, length_scale=jnp.array([test_length_scale]))
model_params = ModelParams(noise_scale=noise_scale, kernel_params=kernel_params)

# Generate samples
print("Generating samples...")
samples_vanilla = feature_vec_prod_vanilla(x, eps, model_params, feature_params, vec)
samples_sym = feature_vec_prod_sym(x, eps, model_params, feature_params, vec)
jax.block_until_ready(samples_vanilla)
jax.block_until_ready(samples_sym)

print("Computing mean and std of samples...")
# compute mean and std of samples
mean_vanilla = jnp.mean(samples_vanilla, axis=None)
std_vanilla = jnp.std(samples_vanilla, axis=None)

mean_sym = jnp.mean(samples_sym, axis=None)
std_sym = jnp.std(samples_sym, axis=None)

print(f"Vanilla mean: {mean_vanilla:.2f}, std: {std_vanilla:.2f}")
print(f"Symmetric mean: {mean_sym:.2f}, std: {std_sym:.2f}")

print("Computing kernel matrix...")
# Compute true kernel matrix with jitter for stability
jitter = 1e-6
K = matern32_kernel_fn(x, x, kernel_params)
K = K + (noise_scale**2 + jitter) * jnp.eye(n_points)

print("Computing Cholesky decomposition...")
# Compute Cholesky decomposition for stable likelihood computation
L = jnp.linalg.cholesky(K)


@jax.jit
def compute_log_likelihood(L, sample):
    # Solve L @ L.T @ x = sample for x using Cholesky
    z = linalg.solve_triangular(L, sample, lower=True)
    log_det = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (jnp.sum(z**2) + log_det + n_points * jnp.log(2 * jnp.pi))


"""# Initialize log-likelihood accumulators
ll_vanilla = jnp.zeros(n_samples_ll)
ll_sym = jnp.zeros(n_samples_ll)

# Compute log-likelihood for each sample
print("Computing log-likelihoods...")
for i in range(n_samples_ll):
    ll_vanilla = ll_vanilla.at[i].set(compute_log_likelihood(L, samples_vanilla[:, i]))
    ll_sym = ll_sym.at[i].set(compute_log_likelihood(L, samples_sym[:, i]))"""

compute_log_likelihoods_batch = jax.vmap(compute_log_likelihood, in_axes=(None, 1))
ll_vanilla = compute_log_likelihoods_batch(L, samples_vanilla)
ll_sym = compute_log_likelihoods_batch(L, samples_sym)

# Average log-likelihoods
ll_vanilla_mean = jnp.mean(ll_vanilla)
ll_sym_mean = jnp.mean(ll_sym)

ll_vanilla_std = jnp.std(ll_vanilla)
ll_sym_std = jnp.std(ll_sym)

print(f"\nAverage log-likelihood per sample:")
print(f"Vanilla method: {ll_vanilla_mean:.2f} ± {ll_vanilla_std:.2f}")
print(f"Symmetric method: {ll_sym_mean:.2f} ± {ll_sym_std:.2f}")


# also do t-test
@partial(jax.jit, static_argnums=(2,))
def ttest_ind_jax(a: jnp.ndarray, b: jnp.ndarray, equal_var: bool = True):
    n, m = a.shape[0], b.shape[0]
    # means
    ma, mb = a.mean(), b.mean()
    # (unbiased) variances
    va = jnp.sum((a - ma) ** 2) / (n - 1)
    vb = jnp.sum((b - mb) ** 2) / (m - 1)

    if equal_var:
        # pooled variance
        sp2 = ((n - 1) * va + (m - 1) * vb) / (n + m - 2)
        se = jnp.sqrt(sp2 * (1 / n + 1 / m))
        df = n + m - 2
    else:
        # Welch’s approximation
        se = jnp.sqrt(va / n + vb / m)
        df = (va / n + vb / m) ** 2 / ((va / n) ** 2 / (n - 1) + (vb / m) ** 2 / (m - 1))

    t_stat = (ma - mb) / se
    # two-sided p via normal approx; for small df you could use jax.scipy.stats.t.cdf
    p_val = 2 * (1 - jax.scipy.special.ndtr(jnp.abs(t_stat)))
    return t_stat, p_val


t, p = ttest_ind_jax(ll_sym, ll_vanilla, equal_var=True)
print(f"T-statistic: {t:.2f}, p-value: {p:.2f}")
