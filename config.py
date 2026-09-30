# config.py
import os


# --- Per-user filesystem paths ---

_USER_PATHS = {
    'username': dict(
        datasets_path='/dataset/path/',
        artifacts_path='/artifacts/path/',
    ),
}

def _resolve_user_paths():
    pwd = os.getcwd()
    for user, paths in _USER_PATHS.items():
        if user in pwd:
            return paths
    raise RuntimeError(f'Unknown user in path: {pwd}')


def get_run_dir(config):
    """<artifacts_path>/<agent_name>_<env_name>/seed_<seed>."""
    job_name = f"{config['agent']['agent_name']}_{config['env_name']}"
    return os.path.join(config['artifacts_path'], job_name, f"seed_{config['seed']}")


# --- Top-level config ---

def get_config(agent_name, env_name):
    paths = _resolve_user_paths()

    agent_config = {
        # --- Core (HIQL defaults) ---
        'agent_name': agent_name,
        'lr': 3e-4,
        'batch_size': 1024,
        'actor_hidden_dims': (512, 512, 512),
        'value_hidden_dims': (512, 512, 512),
        'layer_norm': True,
        'discount': 0.99,
        'tau': 0.005,
        'expectile': 0.7,
        'const_std': True,
        'discrete': False,
        'encoder': None,
        'p_aug': 0.0,
        'frame_stack': None,
        # --- HIQL ---
        'low_alpha': 3.0,
        'high_alpha': 3.0,
        'subgoal_steps': 25,
        'rep_dim': 10,
        'low_actor_rep_grad': False,
        # --- GITA ---
        'abstraction_factor': 5,
        'abstraction_factors': (3, 5, 8, 13, 21),
        'abstraction_hidden_dims': (32,),
        'use_film': True,
        'geo_gamma': 0.98,
        # -- DATASET --
        'value_p_curgoal': 0.2,
        'value_p_trajgoal': 0.5,
        'value_p_randomgoal': 0.3,
        'value_geom_sample': True,
        'actor_p_curgoal': 0.0,
        'actor_p_trajgoal': 1.0,
        'actor_p_randomgoal': 0.0,
        'actor_geom_sample': False,
        'gc_negative': True,
    }

    return {
        'seed': 0,
        'env_name': env_name,
        'artifacts_path': paths['artifacts_path'],
        'datasets_path': paths['datasets_path'],
        'train_steps': 1_000_000,
        'log_interval': 5000,
        'eval_tasks': None,
        'eval_episodes': 50,
        'eval_temperature': 0.1,
        'eval_gaussian': None,
        'eval_on_cpu': False,
        'agent': agent_config,
    }
