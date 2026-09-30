# main.py

import argparse
import csv
import json
import os
import random
from collections import defaultdict

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import jax
import numpy as np
import tqdm
from agents import agents
from config import get_config, get_run_dir
from utils.datasets import HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent


CHECKPOINT_INTERVAL = 100_000
NUM_EVAL_CHECKPOINTS = 3

# Per-environment agent hyperparameters passed by run.sh; everything else comes from config.py.
AGENT_OVERRIDES = {
    'discount': float,
    'geo_gamma': float,
    'batch_size': int,
    'subgoal_steps': int,
    'high_alpha': float,
    'low_alpha': float,
    'actor_p_trajgoal': float,
    'actor_p_randomgoal': float,
    'encoder': str,
    'low_actor_rep_grad': bool,
    'p_aug': float,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent and evaluate its last checkpoints.')
    parser.add_argument('--agent_name', default='gita', choices=sorted(agents))
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_steps', type=int, default=None)
    for key, value_type in AGENT_OVERRIDES.items():
        if value_type is bool:
            parser.add_argument(f'--{key}', action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(f'--{key}', type=value_type)
    return parser.parse_args()


def build_config(args):
    config = get_config(args.agent_name, args.env_name)
    config['seed'] = args.seed
    if args.train_steps is not None:
        config['train_steps'] = args.train_steps
    config['agent'].update(
        {key: getattr(args, key) for key in AGENT_OVERRIDES if getattr(args, key) is not None}
    )
    return config


def write_csv(path, rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_agent(config, env, dataset):
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    example_batch = dataset.sample(1)
    if config['agent']['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent']['agent_name']]
    return agent_class.create(
        config['seed'], example_batch['observations'], example_batch['actions'], config['agent'],
    )


def train(agent, dataset, config, run_dir):
    """Train for config['train_steps'] steps; return the agent and the saved checkpoint steps."""
    train_steps = config['train_steps']
    train_rows = []
    checkpoint_steps = []

    for step in tqdm.trange(1, train_steps + 1, smoothing=0.1, dynamic_ncols=True):
        batch = dataset.sample(config['agent']['batch_size'])
        agent, update_info = agent.update(batch)

        if step % config['log_interval'] == 0:
            train_rows.append({'step': step, **{key: float(value) for key, value in update_info.items()}})

        if step % CHECKPOINT_INTERVAL == 0 or step == train_steps:
            save_agent(agent, run_dir, step)
            write_csv(os.path.join(run_dir, 'train.csv'), train_rows)
            checkpoint_steps.append(step)

    return agent, checkpoint_steps


def evaluate_checkpoints(agent, env, config, run_dir, steps):
    """Evaluate each checkpoint on every task; eval.csv gets one row per task plus an 'overall' row per checkpoint."""
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = config['eval_tasks'] or len(task_infos)
    rows = []

    for step in steps:
        agent, _ = restore_agent(agent, run_dir, step)
        eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0]) if config['eval_on_cpu'] else agent

        task_metrics = defaultdict(list)
        for task_id in tqdm.trange(1, num_tasks + 1, desc=f'eval step {step}', dynamic_ncols=True):
            eval_info = evaluate(
                agent=eval_agent,
                env=env,
                task_id=task_id,
                config=config['agent'],
                num_eval_episodes=config['eval_episodes'],
                eval_temperature=config['eval_temperature'],
                eval_gaussian=config['eval_gaussian'],
            )
            task_name = task_infos[task_id - 1]['task_name']
            rows.append({'checkpoint_step': step, 'task_id': task_id, 'task_name': task_name, **eval_info})
            for key, value in eval_info.items():
                task_metrics[key].append(value)

        overall = {key: float(np.mean(values)) for key, values in task_metrics.items()}
        rows.append({'checkpoint_step': step, 'task_id': 'overall', 'task_name': 'overall', **overall})
        print(f'[step {step}] overall: {json.dumps(overall)}')

    write_csv(os.path.join(run_dir, 'eval.csv'), rows)


def main():
    config = build_config(parse_args())
    run_dir = get_run_dir(config)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"{config['agent']['agent_name']} | {config['env_name']} | seed {config['seed']} -> {run_dir}")

    env, train_data, _ = make_env_and_datasets(
        config['env_name'], config['datasets_path'], frame_stack=config['agent']['frame_stack'],
    )
    dataset = HGCDataset(train_data, config['agent'])
    agent = make_agent(config, env, dataset)

    agent, checkpoint_steps = train(agent, dataset, config, run_dir)
    evaluate_checkpoints(agent, env, config, run_dir, checkpoint_steps[-NUM_EVAL_CHECKPOINTS:])


if __name__ == '__main__':
    main()
