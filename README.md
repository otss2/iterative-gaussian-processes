# Iterative Gaussian Processes
This repository contains the source code for the paper "Improving Linear System Solvers for Hyperparameter Optimisation in Iterative Gaussian Processes" (NeurIPS 2024, [PDF](https://arxiv.org/abs/2405.18457)).

To reproduce the experiments in our paper, please install all the dependencies and use the `train.py` script with the corresponding configurations set in the `config.yaml` file.

Please refer to `toy_example.ipynb` for a simple example of how to use the code in a modular way (for example, to implement your own custom gradient estimator or linear system solver).

If you find this repository useful, please consider citing our paper:
```
@inproceedings{lin2024improving,
    title = {Improving Linear System Solvers for Hyperparameter Optimisation in Iterative Gaussian Processes}, 
    author = {Jihao Andreas Lin and Shreyas Padhy and Bruno Mlodozeniec and Javier Antorán and José Miguel Hernández-Lobato},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2024}
}
```
