# BeyondMimic: Advanced Motion Tracking with Guided Diffusion

**BeyondMimic enables cutting-edge humanoid motion tracking, allowing you to train dynamic, sim-to-real-ready movements with exceptional quality.**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)
[Original Repository](https://github.com/HybridRobotics/whole_body_tracking)

## Key Features

*   **Sim-to-Real Ready Motion:** Train high-quality motions suitable for real-world deployment.
*   **Guided Diffusion-Based Controllers:**  Leverage state-of-the-art control methods for dynamic movements.
*   **LAFAN1 Dataset Support:** Train motion tracking models using the widely used LAFAN1 dataset.
*   **WandB Registry Integration:** Seamlessly manage and load reference motions using Weights & Biases (WandB).

## Overview

BeyondMimic is a versatile humanoid control framework focused on advanced motion tracking. This repository provides the code for training motion tracking models. It allows for training sim-to-real-ready motions using the LAFAN1 dataset without requiring extensive parameter tuning. For sim-to-sim and sim-to-real deployment, see the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller) repository.

## Installation

1.  **Install Isaac Lab v2.1.0:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Conda installation is recommended.
2.  **Clone the Repository:**
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

This section describes how to prepare reference motion data and set up the WandB registry for efficient motion management.

*   **Gather Reference Datasets:**  Obtain reference motion datasets, ensuring you comply with their respective licenses. The project supports datasets like LAFAN1 (Unitree-retargeted), Sidekicks, Christiano Ronaldo celebration (ASAP), and Balance motions (HuB).
    *   LAFAN1 Dataset: [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    *   Sidekicks: [KungfuBot](https://kungfu-bot.github.io/)
    *   Christiano Ronaldo: [ASAP](https://github.com/LeCAR-Lab/ASAP)
    *   Balance motions: [HuB](https://hub-robot.github.io/)
*   **WandB Registry Setup:**
    1.  Log in to your WandB account.
    2.  Create a new registry collection named "Motions" with the artifact type "All Types."
*   **Convert Motions:** Convert retargeted motions to include maximum coordinate information (body pose, velocity, and acceleration) using forward kinematics.
    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This command uploads the processed motion file to the WandB registry.
*   **Test WandB Registry:** Verify proper functionality by replaying the motion in Isaac Sim:
    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
*   **Debugging:**
    *   Ensure `WANDB_ENTITY` is set to your organization name, not your username.
    *   If /tmp is inaccessible, adjust `csv_to_npz.py` (lines 319 and 326) to use an alternative temporary folder.

### Policy Training

*   **Train Policy:** Use the following command to train your motion tracking policy:
    ```bash
    python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
    --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
    --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
    ```

### Policy Evaluation

*   **Play Trained Policy:** Evaluate your trained policy using the following command:
    ```bash
    python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
    ```
    Find the `wandb-run-path` in the run overview (e.g., {your_organization}/{project_name}/{unique_id}). Remember that `run_name` differs from `run_path`.

## Code Structure

Here's a breakdown of the repository's organization:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**:  Contains the core functions for defining the MDP:
    *   `commands.py`: Computes variables from reference motion, robot state, and error calculations.
    *   `rewards.py`: Implements DeepMimic reward functions and smoothing terms.
    *   `events.py`: Defines domain randomization elements.
    *   `observations.py`: Defines observation terms for motion tracking and data collection.
    *   `terminations.py`: Handles early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**:  Environment (MDP) hyperparameter configurations for the tracking task.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters for the tracking task.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings (armature parameters, joint stiffness/damping, action scale).
*   **`scripts`**: Utility scripts for data preprocessing, policy training, and evaluation.