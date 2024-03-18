# RACER: Risk-sensitive Actor Critic with Epistemic Robustness
> Kyle Stachowicz and Sergey Levine
This is the release codebase for [RACER: Risk-sensitive Actor Critic with Epistemic Robustness](arxiv link).

## Setup
Install the base requirements with:
```bash
conda create -n racer python=3.11
pip install -e jaxrl5[train] dmcgym .
pip install -r requirements.txt
```

## Simulation
Run simulation with the following command:
```bash
python scripts/sim/train_online_states.py --config scripts/sim/configs/distributional_limits_config.py --world_name <flat|bumpy> --cvar_risk 0.9
```

## Real robots
Getting things set up from scratch on a new hardware platform is somewhat involved. We use the robot hardware and training workstation setup from [FastRLAP](https://sites.google.com/view/fastrlap), which requires two devices: local compute on the robot (Jetson Orin NX) and a workstation or desktop with GPU for training. See the [offroad-robot-ros](offroad-robot-ros/README.md) directory's instructions for more information.