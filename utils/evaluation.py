from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Split the RNG key before each call to f."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append each value in single_dict to the matching list in dict_of_lists."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def coerce_success(success):
    if isinstance(success, dict):
        return bool(all(success.values()))
    return bool(np.asarray(success).item())


def _run_episode(
    env, actor_fn, config, eval_temperature, eval_gaussian,
    observation, goal, goal_frame, should_render, video_frame_skip,
):
    """Roll out one episode from an already-reset env state.

    Returns a dict with:
        render: list of rendered frames (only populated if should_render).
        info: final step's info dict.
        step: episode length.
    """
    render = []

    done = False
    step = 0
    info = {}
    while not done:
        action, _, _ = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
        action = np.array(action)
        if not config.get('discrete'):
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
            action = np.clip(action, -1, 1)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if should_render and (step % video_frame_skip == 0 or done):
            frame = env.render().copy()
            render.append(np.concatenate([goal_frame, frame], axis=0) if goal_frame is not None else frame)

    return dict(render=render, info=info, step=step)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID passed to the environment.
        config: Agent config dict.
        num_eval_episodes: Number of episodes to evaluate (included in the returned stats).
        num_video_episodes: Number of additional episodes to render (excluded from stats).
        video_frame_skip: Frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Std of Gaussian noise added to actions.

    Returns:
        stats. Rendered video frames are not returned -- nothing consumes them.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    config = config or {}

    stats = defaultdict(list)
    num_successes = 0
    num_completed_eval_episodes = 0
    episode_steps = []

    progress = trange(num_eval_episodes + num_video_episodes)
    progress.set_postfix_str('0/0 (0 success em 0 eps)')
    for i in progress:
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')

        episode = _run_episode(
            env, actor_fn, config, eval_temperature, eval_gaussian,
            observation, goal, goal_frame, should_render, video_frame_skip,
        )

        if should_render:
            continue

        success = coerce_success(episode['info'].get('success', 0.0))

        num_completed_eval_episodes += 1
        num_successes += int(success)
        episode_steps.append(int(episode['step']))
        progress.set_postfix_str(
            f'{num_successes}/{num_completed_eval_episodes} '
            f'({num_successes} success em {num_completed_eval_episodes} eps)'
        )
        episode_info = flatten(episode['info'])
        episode_info['success'] = float(success)
        add_to(stats, episode_info)

    stats = {k: float(np.mean(v)) for k, v in stats.items()}
    if episode_steps:
        stats['episode_steps_mean'] = float(np.mean(episode_steps))
        stats['episode_steps_median'] = float(np.median(episode_steps))
    return stats
