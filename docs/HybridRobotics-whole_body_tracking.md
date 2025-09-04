# BeyondMimic: Advanced Humanoid Motion Tracking for Realistic Robotics

**Achieve state-of-the-art humanoid motion tracking and dynamic control with BeyondMimic, enabling seamless sim-to-real deployment.**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[**[Website]**](https://beyondmimic.github.io/) | [**[Arxiv]**](https://arxiv.org/abs/2508.08241) | [**[Video]**](https://youtu.be/RS_MtKVIAzY) |  [**[GitHub Repo]**](https://github.com/HybridRobotics/whole_body_tracking)

## Overview

BeyondMimic is a powerful framework for dynamic humanoid motion tracking, providing high-quality motion reproduction and test-time control. This repository provides the tools for training motion tracking models, specifically focusing on the LAFAN1 dataset and includes guided diffusion-based controllers.

For sim-to-sim and sim-to-real deployment, refer to the  [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Key Features

*   **Sim-to-Real Readiness:** Train motion tracking models ready for real-world deployment.
*   **LAFAN1 Dataset Support:** Train on the LAFAN1 dataset without extensive parameter tuning.
*   **WandB Registry Integration:** Seamlessly manage and load reference motions using WandB.
*   **Modular Code Structure:** Well-organized code for easy navigation and extension.
*   **Dynamic Motion Control:** Provides highly dynamic motion tracking with state-of-the-art motion quality.

## Installation

### Prerequisites

*   **Isaac Lab v2.1.0:**  Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).  We recommend using the conda installation.
*   **Python 3.10**

### Steps

1.  **Clone the Repository:**
    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git
    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```

2.  **Get Robot Description Files:**
    ```bash
    cd whole_body_tracking
    # Replace 'whole_body_tracking' with your desired extension name (in files/directories)
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```

3.  **Install the Library:** Using a Python interpreter with Isaac Lab installed:
    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

This section details how to prepare motion data using the WandB registry.

1.  **Gather Reference Datasets:**
    *   Unitree-retargeted LAFAN1 Dataset: [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    *   Sidekicks: [KungfuBot](https://kungfu-bot.github.io/)
    *   Christiano Ronaldo Celebration: [ASAP](https://github.com/LeCAR-Lab/ASAP)
    *   Balance Motions: [HuB](https://hub-robot.github.io/)

2.  **WandB Registry Setup:**
    *   Log in to your WandB account.
    *   Create a new registry collection named "Motions" with artifact type "All Types."

3.  **Convert Motions:**
    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This command converts motion data to a format suitable for the WandB registry.

4.  **Test Registry:**
    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
    Verify the motion playback in Isaac Sim.

5.  **Debugging Tips:**
    *   Set `WANDB_ENTITY` to your organization name, not your personal username.
    *   If `/tmp` is inaccessible, adjust `csv_to_npz.py` (lines 319 & 326) to use a different temporary directory.

### Policy Training

Train the motion tracking policy:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Evaluate a trained policy:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```
Find the `wandb_path` in the WandB run overview (e.g., `{your_organization}/{project_name}/{unique_identifier}`).

## Code Structure

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Contains MDP definition, including:
    *   `commands.py`: Computes variables from motion data, robot state, and error calculations.
    *   `rewards.py`: Implements reward functions.
    *   `events.py`: Implements domain randomization.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`: Defines early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment (MDP) configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings.
*   **`scripts`**: Utility scripts for data preprocessing, training, and evaluation.