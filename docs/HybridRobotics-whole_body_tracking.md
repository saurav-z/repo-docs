# BeyondMimic: Train Dynamic Motion Tracking with State-of-the-Art Quality

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[**[GitHub Repository]**](https://github.com/HybridRobotics/whole_body_tracking) | [[Website]](https://beyondmimic.github.io/) | [[Arxiv]](https://arxiv.org/abs/2508.08241) | [[Video]](https://youtu.be/RS_MtKVIAzY)

BeyondMimic offers a powerful framework for creating highly dynamic and realistic humanoid motion tracking, ready for both real-world deployment and advanced control.

## Key Features

*   **Sim-to-Real Ready Motion:** Train motion tracking models ready for deployment with the LAFAN1 dataset.
*   **WandB Registry Integration:** Streamline motion management and loading with WandB for efficient data handling.
*   **Flexible Policy Training & Evaluation:** Train and evaluate policies using provided scripts.
*   **Modular Code Structure:**  Well-organized codebase for easy customization and expansion.
*   **State-of-the-Art Motion Quality:** Achieve highly dynamic motion tracking with advanced control techniques.

## Overview

This repository provides the code for training motion tracking models within the BeyondMimic framework. It focuses on training sim-to-real-ready motion using the LAFAN1 dataset. For deployment in simulation or real-world environments, refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller) repository.

## Installation

1.  **Install Isaac Lab v2.1.0:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), preferably using conda for easy Python script execution.
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

This section details how to manage and prepare motion data using the WandB registry.

1.  **Gather Datasets:**  Obtain reference motion datasets (following original licenses).  Example datasets include:

    *   Unitree-retargeted LAFAN1 Dataset ([HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
    *   Sidekicks ([KungfuBot](https://kungfu-bot.github.io/))
    *   Christiano Ronaldo celebration ([ASAP](https://github.com/LeCAR-Lab/ASAP))
    *   Balance motions ([HuB](https://hub-robot.github.io/))
2.  **WandB Registry Setup:**

    *   Log in to your WandB account.
    *   In the "Core" section, select "Registry".
    *   Create a new registry collection named "Motions" with the artifact type "All Types."

3.  **Convert and Upload Motion Data:** Convert retargeted motions to include maximum coordinate information (pose, velocity, and acceleration) via forward kinematics:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This command automatically uploads the processed `.npz` motion file to the WandB registry.
4.  **Test Registry Integration:** Verify the WandB registry by replaying the motion in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

5.  **Debugging Tips:**
    *   Set `WANDB_ENTITY` to your organization name, not your personal username.
    *   If `/tmp` is inaccessible, modify `csv_to_npz.py` (lines 319 and 326) to use a different temporary directory.

### Policy Training

Train your motion tracking policies using:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Evaluate your trained policies:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The `wandb-run-path` can be found in your WandB run overview.  It follows the format `{your_organization}/{project_name}/` followed by an 8-character identifier.  Note that the `run_name` and `run_path` are distinct.

## Code Structure

The code is organized for modularity and ease of development:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Contains the core MDP functions:
    *   `commands.py`:  Calculates variables from motion, robot state, and error.
    *   `rewards.py`:  Implements reward functions and smoothing.
    *   `events.py`:  Provides domain randomization terms.
    *   `observations.py`: Defines observation terms.
    *   `terminations.py`: Defines early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**:  Environment configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**:  PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings (armature parameters, joint calculations, etc.).
*   **`scripts`**: Utility scripts for data preprocessing, training, and evaluation.