# BeyondMimic: Dynamic Humanoid Motion Tracking and Control

**BeyondMimic empowers dynamic motion tracking and offers advanced control capabilities, enabling state-of-the-art motion quality for real-world applications.**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[Website](https://beyondmimic.github.io/) | [Arxiv](https://arxiv.org/abs/2508.08241) | [Video](https://youtu.be/RS_MtKVIAzY) | [Original Repository](https://github.com/HybridRobotics/whole_body_tracking)

## Key Features

*   **Sim-to-Real Ready Motion Training:** Train dynamic motions for real-world deployment using the LAFAN1 dataset without parameter tuning.
*   **WandB Registry Integration:** Utilizes WandB registry for streamlined motion data management and automatic loading.
*   **Versatile Humanoid Control Framework:** Provides a robust framework for dynamic motion tracking and advanced control.
*   **Guided Diffusion-Based Controllers:** Implements steerable test-time control using diffusion-based controllers.

## Overview

This repository provides the code for motion tracking training within the BeyondMimic framework. It enables the creation of sim-to-real-ready motions, optimized for real-world deployment. The framework utilizes advanced control techniques, including guided diffusion-based controllers, to achieve superior motion quality.

For sim-to-sim and sim-to-real deployment, see the  [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

1.  **Install Isaac Lab:** Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), using the conda installation is recommended.
2.  **Clone the Repository:** Clone this repository separately from your Isaac Lab installation:

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```

3.  **Get Robot Description Files:** Pull the robot description files from GCS:

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```

4.  **Install the Library:**

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

Leverage the WandB registry to store and load reference motions automatically. Retargeted motions should use generalized coordinates only.

*   **Gather Datasets:** Acquire the reference motion datasets (according to their licenses):

    *   Unitree-retargeted LAFAN1 Dataset ([HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
    *   Sidekicks ([KungfuBot](https://kungfu-bot.github.io/))
    *   Christiano Ronaldo celebration ([ASAP](https://github.com/LeCAR-Lab/ASAP))
    *   Balance motions ([HuB](https://hub-robot.github.io/))

*   **Create WandB Registry:** Log in to your WandB account and create a new registry collection named "Motions" with the artifact type "All Types".

*   **Convert Motions:** Convert retargeted motions to include the maximum coordinates information via forward kinematics:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This automatically uploads the processed motion file to the WandB registry.

*   **Test Registry:** Verify functionality by replaying motions in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

*   **Debugging:**
    *   Ensure `WANDB_ENTITY` is set to your organization name, not your personal username.
    *   If the `/tmp` folder isn't accessible, update `csv_to_npz.py` (lines 319 & 326) to use a temporary directory of your choice.

### Policy Training

*   **Train the Policy:**

    ```bash
    python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
    --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
    --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
    ```

### Policy Evaluation

*   **Play the Trained Policy:**

    ```bash
    python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
    ```

    Find the WandB run path in the run overview, formatted as `{your_organization}/{project_name}/` followed by an 8-character identifier. Note that `run_name` is different from `run_path`.

## Code Structure

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Defines the MDP functions:
    *   `commands.py`: Computes variables from motion, robot state, and error calculations.
    *   `rewards.py`: Implements DeepMimic reward functions and smoothing.
    *   `events.py`: Implements domain randomization terms.
    *   `observations.py`: Implements observation terms for motion tracking.
    *   `terminations.py`: Implements early terminations.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment (MDP) hyperparameter configuration.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.

*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings, including armature parameters.

*   **`scripts`**: Utility scripts for motion data preprocessing, training, and policy evaluation.