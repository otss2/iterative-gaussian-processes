from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from mll_optimiser import fit
from config import TrainConfig
from data import get_dataset, subsample
import numpy as np
import os
import kernels
import gradient_estimators
import linear_solvers
import hydra
import omegaconf
import wandb


def get_wandb_name(cfg: TrainConfig):
    wandb_name = f"{cfg.dataset_name}_{cfg.dataset_split}"

    if "matern" in cfg.kernel_name:
        wandb_name += f"_m{cfg.kernel_name[-2:]}"
    elif cfg.kernel_name == "rbf":
        wandb_name += "_rbf"
    else:
        raise ValueError(f"Unknown kernel {cfg.kernel_name}")

    wandb_name += f"_lr{cfg.learning_rate:.0e}"

    if cfg.estimator_name == "exact":
        wandb_name += "_exact"
    else:
        if cfg.estimator_name == "standard":
            wandb_name += "_sd"
        elif cfg.estimator_name == "pathwise":
            wandb_name += "_pw"
        else:
            raise ValueError(f"Unknown estimator name {cfg.estimator_name}")

        if cfg.warm_start:
            wandb_name += "_ws"

        wandb_name += f"_s{cfg.n_samples}"
        wandb_name += f"_{cfg.solver_name}"

        if cfg.solver_name != "chol":
            if cfg.rtol_y > 0.0 or cfg.rtol_z > 0.0:
                wandb_name += f"_rty{cfg.rtol_y:.3f}"
                wandb_name += f"_rtz{cfg.rtol_z:.3f}"
            if cfg.max_epochs > 0:
                wandb_name += f"_maxep{cfg.max_epochs}"

            wandb_name += f"_bs{cfg.batch_size}"

    return wandb_name


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: TrainConfig):
    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key

    # Get the config source path from Hydra
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    # Get the full config path from the runtime config
    try:
        run_name = hydra_cfg.runtime.choices.exp1
    except:
        run_name = get_wandb_name(cfg)
    print(f"\nUsing run name: {run_name}")

    checkpoint_path = f"./checkpoints/{run_name}.npy"

    if cfg.checkpoint_interval > 0 and os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
        run_id = checkpoint["run_id"]
        print(f"Resuming run at i = {checkpoint['step']} ...")
    else:
        if cfg.checkpoint_interval > 0:
            print("No checkpoint found, starting new run...")
        else:
            print("Checkpointing disabled, starting new run...")
        checkpoint = None
        run_id = wandb.util.generate_id()

    with wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        settings=wandb.Settings(start_method="thread"),
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode="online" if cfg.log_wandb else "disabled",
        name=run_name,
        id=run_id,
        resume="allow",
    ) as run:
        wandb_cfg = omegaconf.OmegaConf.create(wandb.config.as_dict())
        cfg = omegaconf.OmegaConf.merge(cfg, wandb_cfg)
        _main(cfg, checkpoint, run_name, run_id)


def _main(cfg: TrainConfig, checkpoint: dict, run_name: str, run_id: str):
    train_ds, test_ds = get_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.n_subsample > 0:
        print(f"Subsampling dataset with n_subsample = {cfg.n_subsample}, subsample_seed = {cfg.subsample_seed} ...")
        train_ds = subsample(train_ds, cfg.subsample_seed, cfg.n_subsample)

    kernel_fn = getattr(kernels, f"{cfg.kernel_name}_kernel_fn")
    kernel_grad_fn = getattr(kernels, f"{cfg.kernel_name}_kernel_grad_fn")
    feature_params_fn = getattr(kernels, f"{cfg.kernel_name}_feature_params_fn")
    estimator = getattr(gradient_estimators, f"{cfg.estimator_name}_gradient")
    solver = getattr(linear_solvers, f"{cfg.solver_name}_solver")

    checkpoint = fit(train_ds, test_ds, kernel_fn, kernel_grad_fn, feature_params_fn, estimator, solver, cfg, checkpoint)
    checkpoint["run_id"] = run_id

    print("noise_scale: ", checkpoint["model_params"].noise_scale)
    print("signal_scale: ", checkpoint["model_params"].kernel_params.signal_scale)
    print("length_scale: ", checkpoint["model_params"].kernel_params.length_scale)

    if cfg.checkpoint_interval > 0:
        print(f"Saving checkpoint to ./checkpoints/{run_name}.npy")
        np.save(f"./checkpoints/{run_name}.npy", checkpoint)
        print("Done!")


if __name__ == "__main__":
    #jax_config.update("jax_disable_jit", True)
    main()
