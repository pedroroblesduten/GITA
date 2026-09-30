#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

SEED=${SEED:-0}

# pointmaze-large-navigate-v0 (GITA)
python main.py --agent_name=gita --env_name=pointmaze-large-navigate-v0 --seed=${SEED} --discount=0.995 --geo_gamma=0.995
# pointmaze-giant-navigate-v0 (GITA)
python main.py --agent_name=gita --env_name=pointmaze-giant-navigate-v0 --seed=${SEED} --discount=0.99 --geo_gamma=0.99
# pointmaze-large-stitch-v0 (GITA)
python main.py --agent_name=gita --env_name=pointmaze-large-stitch-v0 --seed=${SEED} --actor_p_randomgoal=0.5 --actor_p_trajgoal=0.5 --discount=0.98 --geo_gamma=0.98
# pointmaze-giant-stitch-v0 (GITA)
python main.py --agent_name=gita --env_name=pointmaze-giant-stitch-v0 --seed=${SEED} --actor_p_randomgoal=0.5 --actor_p_trajgoal=0.5 --discount=0.99 --geo_gamma=0.99

# antmaze-large-navigate-v0 (GITA)
python main.py --agent_name=gita --env_name=antmaze-large-navigate-v0 --seed=${SEED} --discount=0.98 --geo_gamma=0.98
# antmaze-giant-navigate-v0 (GITA)
python main.py --agent_name=gita --env_name=antmaze-giant-navigate-v0 --seed=${SEED} --discount=0.98 --geo_gamma=0.98
# antmaze-large-stitch-v0 (GITA)
python main.py --agent_name=gita --env_name=antmaze-large-stitch-v0 --seed=${SEED} --actor_p_randomgoal=0.5 --actor_p_trajgoal=0.5 --discount=0.98 --geo_gamma=0.98
# antmaze-giant-stitch-v0 (GITA)
python main.py --agent_name=gita --env_name=antmaze-giant-stitch-v0 --seed=${SEED} --actor_p_randomgoal=0.5 --actor_p_trajgoal=0.5 --discount=0.98 --geo_gamma=0.98
# antmaze-large-explore-v0 (GITA)
python main.py --agent_name=gita --env_name=antmaze-large-explore-v0 --seed=${SEED} --actor_p_randomgoal=1.0 --actor_p_trajgoal=0.0 --discount=0.99 --geo_gamma=0.99 --high_alpha=10.0 --low_alpha=10.0

# humanoidmaze-large-navigate-v0 (GITA)
python main.py --agent_name=gita --env_name=humanoidmaze-large-navigate-v0 --seed=${SEED} --discount=0.99 --geo_gamma=0.99 --subgoal_steps=100
# humanoidmaze-giant-navigate-v0 (GITA)
python main.py --agent_name=gita --env_name=humanoidmaze-giant-navigate-v0 --seed=${SEED} --discount=0.99 --geo_gamma=0.99 --subgoal_steps=100
# humanoidmaze-large-stitch-v0 (GITA)
python main.py --agent_name=gita --env_name=humanoidmaze-large-stitch-v0 --seed=${SEED} --actor_p_randomgoal=0.5 --actor_p_trajgoal=0.5 --discount=0.995 --geo_gamma=0.995 --subgoal_steps=100
# humanoidmaze-giant-stitch-v0 (GITA)
python main.py --agent_name=gita --env_name=humanoidmaze-giant-stitch-v0 --seed=${SEED} --actor_p_randomgoal=0.5 --actor_p_trajgoal=0.5 --discount=0.995 --geo_gamma=0.995 --subgoal_steps=100

# visual-antmaze-large-navigate-v0 (GITA)
python main.py --agent_name=gita --env_name=visual-antmaze-large-navigate-v0 --seed=${SEED} --train_steps=500000 --batch_size=256 --discount=0.98 --geo_gamma=0.98 --encoder=impala_small --high_alpha=3.0 --low_actor_rep_grad --low_alpha=3.0
# visual-antmaze-giant-navigate-v0 (GITA)
python main.py --agent_name=gita --env_name=visual-antmaze-giant-navigate-v0 --seed=${SEED} --train_steps=500000 --batch_size=256 --discount=0.98 --geo_gamma=0.98 --encoder=impala_small --high_alpha=3.0 --low_actor_rep_grad --low_alpha=3.0
# visual-antmaze-large-stitch-v0 (GITA)
python main.py --agent_name=gita --env_name=visual-antmaze-large-stitch-v0 --seed=${SEED} --train_steps=500000 --actor_p_randomgoal=0.5 --actor_p_trajgoal=0.5 --batch_size=256 --discount=0.98 --geo_gamma=0.98 --encoder=impala_small --high_alpha=3.0 --low_actor_rep_grad --low_alpha=3.0
# visual-antmaze-giant-stitch-v0 (GITA)
python main.py --agent_name=gita --env_name=visual-antmaze-giant-stitch-v0 --seed=${SEED} --train_steps=500000 --actor_p_randomgoal=0.5 --actor_p_trajgoal=0.5 --batch_size=256 --discount=0.98 --geo_gamma=0.98 --encoder=impala_small --high_alpha=3.0 --low_actor_rep_grad --low_alpha=3.0

# cube-single-play-v0 (GITA)
python main.py --agent_name=gita --env_name=cube-single-play-v0 --seed=${SEED} --discount=0.98 --geo_gamma=0.98 --high_alpha=3.0 --low_alpha=3.0 --subgoal_steps=10
# cube-double-play-v0 (GITA)
python main.py --agent_name=gita --env_name=cube-double-play-v0 --seed=${SEED} --discount=0.97 --geo_gamma=0.97 --high_alpha=3.0 --low_alpha=3.0 --subgoal_steps=10
# scene-play-v0 (GITA)
python main.py --agent_name=gita --env_name=scene-play-v0 --seed=${SEED} --discount=0.97 --geo_gamma=0.97 --high_alpha=3.0 --low_alpha=3.0 --subgoal_steps=10

# visual-cube-single-noisy-v0 (GITA)
python main.py --agent_name=gita --env_name=visual-cube-single-noisy-v0 --seed=${SEED} --train_steps=500000 --batch_size=256 --encoder=impala_small --high_alpha=3.0 --low_actor_rep_grad --low_alpha=3.0 --p_aug=0.5 --subgoal_steps=10
# visual-cube-double-noisy-v0 (GITA)
python main.py --agent_name=gita --env_name=visual-cube-double-noisy-v0 --seed=${SEED} --train_steps=500000 --batch_size=256 --encoder=impala_small --high_alpha=3.0 --low_actor_rep_grad --low_alpha=3.0 --p_aug=0.5 --subgoal_steps=10
# visual-scene-noisy-v0 (GITA)
python main.py --agent_name=gita --env_name=visual-scene-noisy-v0 --seed=${SEED} --train_steps=500000 --batch_size=256 --encoder=impala_small --high_alpha=3.0 --low_actor_rep_grad --low_alpha=3.0 --p_aug=0.5 --subgoal_steps=10
