# BeyondMimic: Advanced Humanoid Motion Tracking for Sim-to-Real Applications

**Achieve cutting-edge humanoid motion tracking with BeyondMimic, a powerful framework enabling dynamic motion capture and seamless sim-to-real deployment.**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

[View the original repository on GitHub](https://github.com/HybridRobotics/whole_body_tracking)

## Key Features

*   **Sim-to-Real Ready:** Train dynamic motion tracking models easily for real-world deployment.
*   **State-of-the-Art Motion Quality:** Experience high-fidelity motion tracking.
*   **Guided Diffusion-Based Controllers:**  Utilize steerable test-time control.
*   **LAFAN1 Dataset Compatibility:** Train sim-to-real-ready motion directly from the LAFAN1 dataset without parameter tuning.
*   **WandB Registry Integration:** Leverage the WandB registry for streamlined motion data management.

## Overview

BeyondMimic is a versatile framework designed for advanced humanoid control, focusing on dynamic motion tracking with exceptional motion quality suitable for real-world applications. It utilizes guided diffusion-based controllers for test-time control. This repository focuses specifically on the motion tracking training aspects of BeyondMimic, allowing you to train sim-to-real-ready motions directly from the LAFAN1 dataset.

For sim-to-sim and sim-to-real deployment, refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

This section provides steps for setting up the environment to use BeyondMimic.

1.  **Install Isaac Lab:**  Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), using conda is recommended.

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

4.  **Install the Library:**

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking Workflow

The following outlines the process for managing, training, and evaluating motion tracking models within BeyondMimic.

### Motion Preprocessing & Registry Setup

This step is crucial for organizing and preparing motion data using the WandB registry.

1.  **Gather Datasets:** Collect reference motion datasets (following their respective licenses). Supported datasets include:
    *   Unitree-retargeted LAFAN1 Dataset ([HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
    *   Sidekicks ([KungfuBot](https://kungfu-bot.github.io/))
    *   Christiano Ronaldo celebration ([ASAP](https://github.com/LeCAR-Lab/ASAP))
    *   Balance motions ([HuB](https://hub-robot.github.io/))

2.  **WandB Registry Configuration:**

    *   Log in to your WandB account and access the Registry.
    *   Create a new registry collection named "Motions" with the artifact type "All Types."

3.  **Motion Conversion:** Convert retargeted motions into a format that includes body pose, velocity, and acceleration via forward kinematics.

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This script will upload the processed motion to the WandB registry.

4.  **Testing:** Verify the registry setup by replaying a motion in Isaac Sim.

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

5.  **Troubleshooting:**
    *   Ensure `WANDB_ENTITY` is set to your organization name.
    *   If `/tmp` is inaccessible, modify `csv_to_npz.py` (lines 319 & 326) to specify an alternative temporary folder.

### Policy Training

Train your motion tracking policy with the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Evaluate your trained policy:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

Find the `wandb_path` in the run overview (e.g.,  `{your_organization}/{project_name}/{unique_id}`).

## Code Structure

The code is organized for modularity and ease of use:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`:**  Contains the core MDP components:
    *   `commands.py`: Command library for computing variables from reference motions.
    *   `rewards.py`:  DeepMimic reward and smoothing functions.
    *   `events.py`: Domain randomization terms.
    *   `observations.py`: Observation terms.
    *   `terminations.py`: Termination conditions.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`:** Environment configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`:** PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`:** Robot-specific settings.
*   **`scripts`:**  Utility scripts for preprocessing data, training, and evaluation.