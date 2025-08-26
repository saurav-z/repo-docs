# BeyondMimic: State-of-the-Art Humanoid Motion Tracking

**BeyondMimic offers a powerful framework for dynamic humanoid motion tracking, delivering high-quality, sim-to-real-ready results for advanced robotics applications.**  ([Original Repository](https://github.com/HybridRobotics/whole_body_tracking))

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

*   **Sim-to-Real Ready:** Train robust motion tracking models for real-world deployment.
*   **High-Quality Motion:** Achieve state-of-the-art motion quality.
*   **Guided Diffusion-Based Controllers:** Utilize steerable test-time control.
*   **LAFAN1 Dataset Compatibility:** Train motion tracking models with ease on the LAFAN1 dataset, without extensive parameter tuning.
*   **WandB Registry Integration:** Leverage the WandB registry for streamlined motion data management.

## Overview

BeyondMimic is a versatile humanoid control framework designed for dynamic motion tracking. This repository focuses on training motion tracking models, enabling you to create sim-to-real-ready motions within the LAFAN1 dataset, with minimal parameter adjustments. For deployment, refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

Follow these steps to get started:

1.  **Install Isaac Lab v2.1.0:**  Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), preferably using the conda installation method.

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

4.  **Install the Library:**  Using a Python interpreter with Isaac Lab installed:
    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

This section describes how to prepare your motion data using a WandB registry.

*   **Gather Reference Motion Datasets:**  Ensure you have the necessary datasets, following their respective licenses.  The project uses datasets like LAFAN1 (retargeted), Sidekicks, Christiano Ronaldo celebration (ASAP), and Balance motions (HuB).

*   **WandB Registry Setup:**
    1.  Log in to your WandB account.
    2.  Access the Registry under Core on the left side.
    3.  Create a new registry collection named "Motions" with the artifact type "All Types".

*   **Convert and Upload Motions:**  Convert your retargeted motions to include maximum coordinate information (body pose, velocity, and acceleration).

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This script automatically uploads the processed motion file to the WandB registry.

*   **Test Registry Integration:** Replay your motion within Isaac Sim to confirm the setup:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

*   **Debugging Tips:**
    *   Set `WANDB_ENTITY` to your organization name.
    *   If the `/tmp` folder is inaccessible, modify `csv_to_npz.py` (lines 319 & 326) to use a different temporary directory.

### Policy Training

Train your motion tracking policy with the following command:

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

Find your `wandb_path` in the WandB run overview (format: `{your_organization}/{project_name}/{unique_8-character-identifier}`).  Remember, `run_name` is different from `run_path`.

## Code Structure

The code is organized to facilitate modularity and ease of use:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`:** Core functions for the MDP:
    *   `commands.py`:  Computes variables from reference motion, robot state, and error calculations.
    *   `rewards.py`: Implements DeepMimic reward functions.
    *   `events.py`: Implements domain randomization terms.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`: Implements early terminations and timeouts.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`:** Environment configuration.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`:** PPO hyperparameters.

*   **`source/whole_body_tracking/whole_body_tracking/robots`:** Robot-specific settings (armature parameters, joint stiffness/damping, action scaling).

*   **`scripts`:** Utility scripts for preprocessing, training, and evaluation.