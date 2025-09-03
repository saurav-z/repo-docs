# BeyondMimic: Advanced Humanoid Motion Tracking for Realistic Simulations

BeyondMimic provides cutting-edge motion tracking capabilities, allowing you to train and deploy highly dynamic and realistic humanoid motions. **Train sim-to-real-ready motion directly using the LAFAN1 dataset without any parameter tuning!**

[Visit the original repository for more details](https://github.com/HybridRobotics/whole_body_tracking)

## Key Features

*   **Sim-to-Real Ready:** Directly train motion for real-world deployment.
*   **State-of-the-Art Motion Quality:** Achieve highly dynamic and realistic motion tracking.
*   **Guided Diffusion-Based Controllers:** Utilize steerable test-time control.
*   **LAFAN1 Dataset Compatibility:** Train motions from the LAFAN1 dataset without parameter tuning.
*   **WandB Registry Integration:** Leverage the WandB registry for managing and loading reference motions.
*   **Modular Code Structure:** Designed for easy navigation and expansion by developers.

## Overview

BeyondMimic is a versatile humanoid control framework focused on dynamic motion tracking. This repository contains the code for motion tracking training. It offers a robust solution for creating realistic humanoid movements, with a particular focus on the LAFAN1 dataset.  For sim-to-sim and sim-to-real deployment, please refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

Follow these steps to set up the environment:

1.  **Install Isaac Lab:** Install Isaac Lab v2.1.0 following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Conda installation is recommended.
2.  **Clone the Repository:** Clone this repository outside the Isaac Lab directory using either SSH or HTTPS:

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```
3.  **Fetch Robot Description Files:** Download robot description files from GCS:

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```
4.  **Install the Library:** Using a Python interpreter with Isaac Lab installed, install the library:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking Guide

### Motion Preprocessing & Registry Setup

Utilize the WandB registry to manage and automatically load reference motions.

*   **Gather Datasets:** Acquire reference motion datasets (following original licenses).  The repository uses Unitree retargeted LAFAN1 Dataset (HuggingFace), Sidekicks (KungfuBot), Christiano Ronaldo celebration (ASAP), and Balance motions (HuB).
*   **WandB Registry Setup:**
    *   Log in to your WandB account and access the Registry.
    *   Create a new registry collection named "Motions" with the artifact type "All Types".
*   **Convert Motions:** Convert retargeted motions to include maximum coordinate information using forward kinematics:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This will automatically upload the processed motion to the WandB registry.
*   **Test Registry:** Verify the WandB registry is working by replaying a motion in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
*   **Debugging:**
    *   Set `WANDB_ENTITY` to your organization name.
    *   If /tmp is inaccessible, modify `csv_to_npz.py` (lines 319 & 326) to a temporary directory.

### Policy Training

Train a policy using the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Evaluate the trained policy with:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The WandB run path is found in the run overview and follows the format `{your_organization}/{project_name}/{unique_identifier}`. Note that run_name is different from run_path.

## Code Structure Breakdown

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`:** Core MDP functions:
    *   `commands.py`: Computes variables from reference motions and robot state.
    *   `rewards.py`: Implements reward functions.
    *   `events.py`: Implements domain randomization.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`: Handles early terminations.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment (MDP) configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings.
*   **`scripts`**: Utility scripts for motion data preprocessing, policy training, and evaluation.