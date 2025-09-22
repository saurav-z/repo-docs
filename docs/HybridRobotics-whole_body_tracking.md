# BeyondMimic: Humanoid Motion Tracking with State-of-the-Art Quality

BeyondMimic is a powerful framework for achieving dynamic and high-quality motion tracking on humanoid robots, enabling sim-to-real transfer and steerable control. Access the original repository [here](https://github.com/HybridRobotics/whole_body_tracking).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

## Key Features

*   **Sim-to-Real Ready Motion Training:** Train dynamic motions from the LAFAN1 dataset with minimal parameter tuning.
*   **Guided Diffusion-Based Controllers:** Offers steerable test-time control.
*   **WandB Registry Integration:**  Leverages the WandB registry for streamlined motion data management and automatic loading.
*   **Modular Code Structure:** Designed for ease of use, modification, and expansion.

## Overview

BeyondMimic provides a versatile humanoid control framework designed for highly dynamic motion tracking, with a focus on achieving state-of-the-art motion quality and seamless sim-to-real deployment.  This repository specifically focuses on the motion tracking training aspects of BeyondMimic. For deployment on sim-to-sim and sim-to-real platforms, refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

1.  **Install Isaac Lab:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for Isaac Lab v2.1.0.  Conda installation is recommended.
2.  **Clone the Repository:** Clone this repository outside of your Isaac Lab installation:

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```

3.  **Pull Robot Description Files:**

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```

4.  **Install the Library:** Using a Python interpreter with Isaac Lab installed:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

This section details the use of the WandB registry for motion data management.

*   **Gather Reference Datasets:** Collect motion datasets, adhering to their respective licenses. The repository supports datasets like LAFAN1 (retargeted), Sidekicks, Christiano Ronaldo celebration (ASAP), and balance motions (HuB).
*   **WandB Registry Setup:**
    1.  Log in to your WandB account.
    2.  Create a new registry collection named "Motions" with the artifact type "All Types."
*   **Convert Motions:** Convert retargeted motions to include maximum coordinate information via forward kinematics:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This command uploads the processed motion file to the WandB registry.
*   **Test Motion Replay:** Verify the WandB registry integration by replaying motions in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

*   **Debugging:**
    *   Ensure that `WANDB_ENTITY` is set to your organization name.
    *   If `/tmp` is inaccessible, modify `csv_to_npz.py` (lines 319 & 326) to use an alternative temporary directory.

### Policy Training

Train a policy using the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Evaluate a trained policy using:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The `wandb_path` can be found in the WandB run overview, following the format `{your_organization}/{project_name}/` with an 8-character identifier. Note that `run_name` differs from `run_path`.

## Code Structure Overview

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`:** Core MDP functions:
    *   `commands.py`: Computes variables from reference motion, robot state, and errors.
    *   `rewards.py`: Implements DeepMimic reward functions.
    *   `events.py`: Domain randomization.
    *   `observations.py`: Observation terms for motion tracking and data collection.
    *   `terminations.py`: Early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`:** Environment (MDP) hyperparameter configurations.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`:** PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`:** Robot-specific settings.
*   **`scripts`:** Utility scripts for data preprocessing, training, and evaluation.