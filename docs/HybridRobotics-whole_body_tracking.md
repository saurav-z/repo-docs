# BeyondMimic: State-of-the-Art Humanoid Motion Tracking

**BeyondMimic provides a cutting-edge framework for dynamic humanoid motion tracking, delivering high-quality motion on real-world deployments and steerable test-time control.** ([Original Repo](https://github.com/HybridRobotics/whole_body_tracking))

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

*   **Sim-to-Real Ready:** Train motion for deployment with the LAFAN1 dataset with no parameter tuning.
*   **WandB Registry Integration:** Seamlessly manage and load reference motions using the WandB registry.
*   **Modular Code Structure:** Designed for easy navigation and expansion.
*   **Versatile Motion Tracking:** Achieve highly dynamic motion tracking for humanoid robots.
*   **Guided Diffusion-Based Controllers:** Utilize steerable test-time control.

## Overview

BeyondMimic is a versatile framework focused on dynamic humanoid motion tracking. This repository focuses on the motion tracking training aspects of BeyondMimic, empowering users to generate sim-to-real-ready motion. For sim-to-sim and sim-to-real deployment, refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

1.  **Install Isaac Lab:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for Isaac Lab v2.1.0. Conda installation is recommended.
2.  **Clone the Repository:** Clone this repository outside the Isaac Lab directory:

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```

3.  **Fetch Robot Description Files:**

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```

4.  **Install the Library:** Using a Python interpreter with Isaac Lab installed, run:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

This section outlines the process for managing and loading reference motions using the WandB registry.

*   **Reference Datasets:** Gather reference motion datasets, adhering to their licenses. The project uses datasets from:
    *   LAFAN1 (Unitree-retargeted) - [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    *   Sidekicks - [KungfuBot](https://kungfu-bot.github.io/)
    *   Christiano Ronaldo celebration - [ASAP](https://github.com/LeCAR-Lab/ASAP)
    *   Balance motions - [HuB](https://hub-robot.github.io/)
*   **WandB Registry:**
    1.  Log in to your WandB account.
    2.  Create a new registry collection named "Motions" with the artifact type "All Types" under Core on the left panel.
*   **Convert Motions:** Convert retargeted motions to include maximum coordinates information using forward kinematics:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This uploads the processed motion file to the WandB registry.
*   **Test Registry:** Verify the WandB registry by replaying the motion in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

*   **Debugging:**
    *   Ensure `WANDB_ENTITY` is set to your organization name, not your username.
    *   If the `/tmp` folder is inaccessible, modify `csv_to_npz.py` (lines 319 & 326) to use a temporary folder of your choice.

### Policy Training

*   Train a policy using:

    ```bash
    python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
    --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
    --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
    ```

### Policy Evaluation

*   Play the trained policy using:

    ```bash
    python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
    ```

    Find the WandB run path in the run overview (e.g., `{your_organization}/{project_name}/{unique_8_char_id}`).  Note that `run_name` is distinct from `run_path`.

## Code Structure

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`:** Core MDP definitions:
    *   `commands.py`: Computes variables from reference motions, robot states, and error calculations.
    *   `rewards.py`: Implements DeepMimic reward functions.
    *   `events.py`: Implements domain randomization.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`: Implements early terminations.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`:**  Environment (MDP) hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`:** PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`:** Robot-specific settings (armature parameters, joint calculations, action scaling).
*   **`scripts`:** Utility scripts for motion data preprocessing, training, and evaluation.