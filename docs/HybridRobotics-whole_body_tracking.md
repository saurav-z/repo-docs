# BeyondMimic: Achieve State-of-the-Art Humanoid Motion Tracking for Real-World Deployment

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

**BeyondMimic provides a robust framework for training and deploying high-quality humanoid motion tracking models, enabling dynamic and realistic movement in simulation and the real world.**  Find the original repo [here](https://github.com/HybridRobotics/whole_body_tracking).

## Key Features

*   **Sim-to-Real Readiness:** Train motion tracking models suitable for real-world deployment.
*   **LAFAN1 Dataset Compatibility:** Train motion data from the LAFAN1 dataset, requiring no parameter tuning.
*   **Guided Diffusion-Based Control:** Utilize advanced controllers for steerable test-time control.
*   **WandB Registry Integration:** Seamlessly manage and load reference motions using the Weights & Biases registry.
*   **Modular Code Structure:** Well-organized code for easy navigation and development.

## Overview

BeyondMimic is a versatile humanoid control framework that offers high-quality motion tracking, ensuring realistic movement. This repository focuses on motion tracking training, allowing you to create sim-to-real-ready motions. For deployment, refer to the [motion\_tracking\_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

### Prerequisites

*   Isaac Lab v2.1.0 (Installation Guide: [https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)). Use conda for Python script calls.
*   Python 3.10

### Steps

1.  **Clone the Repository:**

    *   **SSH:**
        ```bash
        git clone git@github.com:HybridRobotics/whole_body_tracking.git
        ```
    *   **HTTPS:**
        ```bash
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

### Motion Preprocessing & Registry Setup

This project leverages the WandB registry for easy motion management.

1.  **Gather Datasets:**  Use reference motion datasets (follow original licenses):

    *   Unitree-retargeted LAFAN1 ([HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
    *   Sidekicks ([KungfuBot](https://kungfu-bot.github.io/))
    *   Cristiano Ronaldo celebration ([ASAP](https://github.com/LeCAR-Lab/ASAP))
    *   Balance motions ([HuB](https://hub-robot.github.io/))

2.  **WandB Setup:**

    *   Log in to your WandB account.
    *   Create a new registry collection named "Motions" with artifact type "All Types".

3.  **Convert Motions:** Convert retargeted motions to include maximum coordinate information.

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This will automatically upload the processed motion file to the WandB registry.

4.  **Test Registry:**

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

5.  **Debugging:**

    *   Set `WANDB_ENTITY` to your organization name, not your username.
    *   If `/tmp` is inaccessible, change `csv_to_npz.py` lines 319 & 326 to an alternate temporary folder.

### Policy Training

1.  **Train Policy:**

    ```bash
    python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
    --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
    --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
    ```

### Policy Evaluation

1.  **Play Trained Policy:**

    ```bash
    python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
    ```

    Find the `wandb_path` in the WandB run overview (e.g., `{your_organization}/{project_name}/`).

## Code Structure

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Core MDP functions:
    *   `commands.py`: Computes variables from reference motion, robot state, and error.
    *   `rewards.py`: DeepMimic reward functions.
    *   `events.py`: Domain randomization.
    *   `observations.py`: Observation terms.
    *   `terminations.py`: Early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment (MDP) hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings.
*   **`scripts`**: Utility scripts for data preprocessing, training, and evaluation.