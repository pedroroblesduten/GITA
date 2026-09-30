# GITA: Learning Multiple Timescales for Goal-Conditioned Reinforcement Learning

Official implementation of **Generalized Implicit Temporal Abstraction (GITA)**, from the paper
*Learning Multiple Timescales for Goal-Conditioned Reinforcement Learning*.

## Overview

In offline goal-conditioned RL, temporal abstraction (treating `k` environment steps as one
transition) keeps value differences between states measurable at long range. No single `k` suits
every state-goal distance, though. A large `k` preserves signal for distant goals but blurs nearby
states; a small `k` does the reverse.

GITA does not commit to a single `k`:

- **One value function for all scales.** The high-level value `V(s, g; k)` is conditioned on the
  abstraction factor `k`. During training, `k ~ Geom(1 - α)` is sampled per example.
- **One policy trained on every scale.** For each subgoal, GITA computes the advantage
  `A(s, s', g; k) = V(s', g; k) - V(s, g; k)` for every `k` in a fixed candidate set `K`. It
  trains a single high-level policy with the averaged AWR weight
  `W = mean_k exp(β · A(s, s', g; k))`. Because the exponential is applied before averaging,
  the scale with the clearest positive advantage dominates each update. This gives an *implicit*
  selection of scale per state-goal pair.
- **No scale at inference.** The policy does not take `k` as input. The low-level policy is
  unchanged from HIQL.

## Installation

This code uses JAX and [OGBench](https://github.com/seohongpark/ogbench) (Python 3.12). It was
developed with `jax 0.8.2`, `flax 0.12.4`, `optax 0.2.6`, `distrax 0.1.7`, and `ogbench 1.2.1`.

Before the first run, set the dataset and output directories in `_USER_PATHS` in
[config.py](config.py).

## Usage

### Train and evaluate on one environment

```shell
python main.py --env_name=antmaze-giant-navigate-v0 --seed=0 --discount=0.98
```

`main.py` trains for 1M steps (500K for pixel-based environments via `--train_steps=500000`),
saving a checkpoint every 100K steps. It then evaluates the last three checkpoints on the
environment's five evaluation tasks, with 50 episodes per task. Outputs are written to
`<artifacts_path>/gita_<env_name>/seed_<seed>/`:

| File | Content |
|---|---|
| `config.json` | Full configuration of the run |
| `train.csv` | Training losses every 5K steps |
| `params_<step>.pkl` | Checkpoints |
| `eval.csv` | Success rate per task and overall, for each evaluated checkpoint |

### Reproduce the paper's environments

[run.sh](run.sh) lists one command per environment with its tuned hyperparameters, in the format
of OGBench's `hyperparameters.sh`, and runs them in sequence:

## Repository structure

```
main.py          training + evaluation entry point
config.py        default hyperparameters and paths
run.sh           per-environment commands
agents/gita.py   GITA agent
agents/hiql.py   HIQL baseline
agents/ota.py    OTA baseline
utils/           datasets, networks, encoders, evaluation, checkpointing
```

