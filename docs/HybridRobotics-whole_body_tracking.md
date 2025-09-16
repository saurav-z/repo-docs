# BeyondMimic: Advanced Humanoid Motion Tracking for Realistic Simulation and Deployment

**BeyondMimic offers cutting-edge motion tracking for humanoid robots, providing state-of-the-art motion quality that's ready for real-world applications.** [Find the original repository here](https://github.com/HybridRobotics/whole_body_tracking).

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

*   **Sim-to-Real Ready:** Train motion tracking models that are readily deployable in real-world scenarios.
*   **State-of-the-Art Motion Quality:** Achieves highly dynamic and realistic humanoid motions.
*   **Guided Diffusion-Based Controllers:**  Offers steerable test-time control using guided diffusion techniques.
*   **LAFAN1 Dataset Compatibility:** Train motion tracking models using the widely used LAFAN1 dataset.
*   **WandB Registry Integration:** Leverages WandB for efficient motion data management and tracking.

## Overview

BeyondMimic is a versatile humanoid control framework designed for high-fidelity motion tracking and control.  This repository focuses on the motion tracking training aspect, allowing you to generate sim-to-real ready motion. For deployment strategies, refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller) repository.

## Installation

**Prerequisites:**

*   Isaac Lab v2.1.0 (install using the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), conda recommended)
*   Python 3.10

**Steps:**

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
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```

3.  **Install the Library:**
    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

This section details how to prepare and train motion tracking models.

### Motion Preprocessing & Registry Setup

This process uses the WandB registry to organize and load reference motions.

1.  **Gather Datasets:** Prepare reference motion datasets (respecting their original licenses).  Supported datasets include the retargeted LAFAN1 dataset, KungfuBot's Sidekicks, Christiano Ronaldo celebration from ASAP, and balance motions from HuB.
2.  **WandB Registry Setup:**
    *   Log in to your WandB account and access "Registry" under "Core".
    *   Create a new registry collection named "Motions" with artifact type "All Types".
3.  **Motion Conversion:** Convert retargeted motions to include maximum coordinates information (body pose, velocity, and acceleration) using forward kinematics:
    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This uploads the processed file to the WandB registry.
4.  **Testing:** Verify registry functionality by replaying a motion in Isaac Sim:
    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
5.  **Debugging:**
    *   Set the `WANDB_ENTITY` environment variable to your organization's name.
    *   If `/tmp` is inaccessible, adjust `csv_to_npz.py` (lines 319 & 326) to use a suitable temporary folder.

### Policy Training

Train your motion tracking policy using the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Play the trained policy:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

*   Locate the WandB run path in the run overview (format: `{your_organization}/{project_name}/` followed by a unique identifier). Note that `run_name` is different from `run_path`.

## Code Structure

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`:**  Contains the core functions for the MDP:
    *   `commands.py`: Computes relevant variables from motion, robot state, and errors.
    *   `rewards.py`: Implements reward functions and smoothing.
    *   `events.py`: Defines domain randomization.
    *   `observations.py`:  Implements observation terms.
    *   `terminations.py`:  Handles early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`:**  MDP environment configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`:**  PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`:**  Robot-specific settings (armature, joint parameters, action scaling).
*   **`scripts`:** Utility scripts for motion preprocessing, policy training, and evaluation.