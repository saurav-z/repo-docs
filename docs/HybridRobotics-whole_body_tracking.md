# BeyondMimic: Advanced Whole-Body Motion Tracking

**Achieve state-of-the-art, sim-to-real-ready motion tracking and steerable control for humanoid robots with BeyondMimic.**  [View the original repository](https://github.com/HybridRobotics/whole_body_tracking).

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

*   **Sim-to-Real Ready Motion:** Train highly dynamic motion tracking models for real-world deployment.
*   **Guided Diffusion-Based Control:** Implement steerable test-time control using diffusion-based controllers.
*   **LAFAN1 Dataset Compatibility:** Train sim-to-real-ready motions using the LAFAN1 dataset without parameter tuning.
*   **WandB Registry Integration:** Utilize the WandB registry for efficient motion data management.
*   **Modular Code Structure:** Navigate and extend the project with a clearly defined code architecture.

## Overview

BeyondMimic is a comprehensive humanoid control framework designed for advanced motion tracking. It delivers state-of-the-art motion quality, facilitating real-world deployment and incorporating steerable control mechanisms. This repository provides the code for motion tracking training, enabling the creation of sim-to-real-ready motions using the LAFAN1 dataset.

For Sim-to-Sim and Sim-to-Real deployment, please refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

1.  **Install Isaac Lab:** Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).  Conda is recommended.
2.  **Clone the Repository:**

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```

3.  **Get Robot Description Files:**

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```

4.  **Install the Library:**  Using a Python interpreter with Isaac Lab installed:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking Guide

### Motion Preprocessing & Registry Setup

This section guides you through preparing motion data and setting up the WandB registry for managing reference motions.

*   **Gather Datasets:**  Obtain reference motion datasets, such as:

    *   Unitree-retargeted LAFAN1 Dataset (HuggingFace)
    *   Sidekicks (KungfuBot)
    *   Christiano Ronaldo celebration (ASAP)
    *   Balance motions (HuB)

    (Follow the original licenses.)

*   **WandB Registry Setup:**
    1.  Log in to your WandB account.
    2.  Navigate to "Registry" under "Core" on the left.
    3.  Create a new registry collection named "Motions" with artifact type "All Types."

*   **Convert Motions:** Convert retargeted motions to include maximum coordinate information (body pose, velocity, and acceleration) using forward kinematics:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This command automatically uploads the processed motion file to the WandB registry.

*   **Test Registry:** Verify the WandB registry setup by replaying the motion in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

*   **Debugging:**
    *   Ensure `WANDB_ENTITY` is set to your organization name.
    *   If `/tmp` is inaccessible, modify `csv_to_npz.py` (lines 319 & 326) to specify an alternative temporary folder.

### Policy Training

Train a policy using the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Play the trained policy with:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The `wandb_path` can be found in the run overview, following the format `{your_organization}/{project_name}/{unique_identifier}`.

## Code Structure

The code is organized for modularity:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**:  Contains the MDP definition:
    *   `commands.py`: Computes variables from reference motion and robot state.
    *   `rewards.py`: Implements reward functions.
    *   `events.py`:  Implements domain randomization.
    *   `observations.py`: Defines observation terms.
    *   `terminations.py`:  Handles early terminations.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific parameters.
*   **`scripts`**: Utility scripts for data preprocessing, training, and evaluation.