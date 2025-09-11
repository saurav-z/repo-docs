# BeyondMimic: Advanced Humanoid Motion Tracking for Realistic Robotics

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[**[Website]**](https://beyondmimic.github.io/) | [**[Arxiv]**](https://arxiv.org/abs/2508.08241) | [**[Video]**](https://youtu.be/RS_MtKVIAzY) | [**[Original Repo]**](https://github.com/HybridRobotics/whole_body_tracking)

BeyondMimic is a cutting-edge framework for creating realistic and dynamic humanoid motion tracking, offering state-of-the-art motion quality for real-world deployment.

**Key Features:**

*   **Sim-to-Real Ready:** Train sim-to-real motion with minimal parameter tuning.
*   **WandB Registry Integration:** Seamlessly manage and load reference motions using WandB.
*   **Guided Diffusion-Based Controllers:** Leverage steerable test-time control for advanced motion capabilities.
*   **Modular Code Structure:** Organized codebase for easy navigation and expansion.
*   **LAFAN1 Dataset Compatibility:** Train on the LAFAN1 dataset for a wide range of motions.

## Overview

This repository provides the motion tracking training code for BeyondMimic, enabling you to create dynamic and realistic humanoid motions.  It utilizes a versatile humanoid control framework, offering state-of-the-art motion quality on real-world deployments, and steerable test-time control with guided diffusion-based controllers. This project makes it simple to train sim-to-real-ready motion using the LAFAN1 dataset, without requiring extensive parameter tuning.  For sim-to-sim and sim-to-real deployment, please refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

To get started, follow these steps:

1.  **Install Isaac Lab:** Install Isaac Lab v2.1.0 by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Conda installation is recommended.

2.  **Clone the Repository:** Clone this repository separately from the Isaac Lab installation:

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```

3.  **Fetch Robot Description Files:** Pull the robot description files from GCS.

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

This section details how to set up the WandB registry for managing reference motions.

*   **Gather Datasets:**  Obtain reference motion datasets. (Follow original licenses)
    *   Unitree-retargeted LAFAN1 Dataset ([HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
    *   Sidekicks ([KungfuBot](https://kungfu-bot.github.io/))
    *   Christiano Ronaldo celebration ([ASAP](https://github.com/LeCAR-Lab/ASAP))
    *   Balance motions ([HuB](https://hub-robot.github.io/))

*   **WandB Registry Setup:**
    1.  Log in to your WandB account.
    2.  Access the Registry under "Core" on the left.
    3.  Create a new registry collection named "Motions" with the artifact type "All Types".

*   **Convert Motions:** Convert retargeted motions to include maximum coordinate information (body pose, velocity, and acceleration).

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This command uploads processed motion files to the WandB registry.

*   **Verify:**  Test the registry by replaying motions in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

*   **Troubleshooting:**
    *   Ensure `WANDB_ENTITY` is set to your organization name.
    *   If `/tmp` is inaccessible, adjust `csv_to_npz.py` (lines 319 & 326) to a suitable temporary folder.

### Policy Training

Train a motion tracking policy using the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Evaluate the trained policy:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The `wandb_path` can be found in the run overview on WandB (e.g.,  `{your_organization}/{project_name}/<unique_identifier>`).  Note the distinction between `run_name` and `run_path`.

## Code Structure

The project's code is organized as follows:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Contains core MDP functions:

    *   `commands.py`:  Calculates variables from reference motion, robot state, and errors.
    *   `rewards.py`: Defines DeepMimic reward and smoothing functions.
    *   `events.py`: Implements domain randomization terms.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`: Implements early terminations and timeouts.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment (MDP) configuration.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.

*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings (armature parameters, joint calculations, and action scaling).

*   **`scripts`**: Utility scripts for motion preprocessing, training, and evaluation.