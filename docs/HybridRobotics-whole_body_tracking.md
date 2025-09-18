# BeyondMimic: Train Dynamic Humanoid Motion Tracking with Ease

**Achieve state-of-the-art motion quality and sim-to-real readiness for humanoid robots with BeyondMimic, a cutting-edge motion tracking framework.**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[Website](https://beyondmimic.github.io/) | [Arxiv](https://arxiv.org/abs/2508.08241) | [Video](https://youtu.be/RS_MtKVIAzY) | [GitHub Repository](https://github.com/HybridRobotics/whole_body_tracking)

## Key Features

*   **Sim-to-Real Ready:** Train motion tracking models easily for real-world deployment.
*   **State-of-the-Art Motion Quality:** Achieve highly dynamic and realistic motion tracking.
*   **Guided Diffusion-Based Controllers:** Utilize steerable test-time control.
*   **LAFAN1 Dataset Compatibility:** Train on LAFAN1 data out-of-the-box without parameter tuning.
*   **WandB Integration:** Leverage WandB registry for motion management and policy tracking.
*   **Modular Code Structure:** Clear organization for easier development and modification.

## Overview

BeyondMimic is a versatile humanoid control framework designed for dynamic motion tracking, offering superior motion quality for real-world applications and advanced control capabilities. This repository provides the necessary code for training motion tracking models, making it easy to generate sim-to-real ready motions using the LAFAN1 dataset. For sim-to-sim and sim-to-real deployment, consult the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

1.  **Install Isaac Lab v2.1.0:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), using conda is recommended.

2.  **Clone the Repository:** Clone this repository outside of your Isaac Lab installation directory:

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

4.  **Install the Library:**

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

BeyondMimic uses the WandB registry for managing motion datasets.

*   **Gather Datasets:**
    *   Unitree-retargeted LAFAN1 Dataset: [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    *   Sidekicks: [KungfuBot](https://kungfu-bot.github.io/)
    *   Christiano Ronaldo Celebration: [ASAP](https://github.com/LeCAR-Lab/ASAP)
    *   Balance motions: [HuB](https://hub-robot.github.io/)
*   **WandB Setup:** Log in to your WandB account and create a registry collection named "Motions" with artifact type "All Types".
*   **Convert Motion Data:** Convert motion data to include maximum coordinate information:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This will automatically upload the processed motion file to the WandB registry.
*   **Test Registry:** Verify functionality in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
*   **Debugging:**
    *   Ensure you export `WANDB_ENTITY` with your organization name.
    *   If /tmp folder is inaccessible, modify `csv_to_npz.py` to use an alternative temporary folder.

### Policy Training

*   Train a policy using the following command:

    ```bash
    python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
    --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
    --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
    ```

### Policy Evaluation

*   Evaluate a trained policy:

    ```bash
    python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
    ```
    Find the WandB run path in the run overview, formatted as `{your_organization}/{project_name}/` followed by an 8-character identifier. Note that `run_name` and `run_path` are distinct.

## Code Structure

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Core MDP functions:
    *   `commands.py`: Reference motion processing, robot state calculations, and error computations.
    *   `rewards.py`: DeepMimic reward implementation and smoothing terms.
    *   `events.py`: Domain randomization terms.
    *   `observations.py`: Motion tracking and data collection observations.
    *   `terminations.py`: Early termination and timeout conditions.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings.
*   **`scripts`**: Utility scripts for data preprocessing, policy training, and evaluation.