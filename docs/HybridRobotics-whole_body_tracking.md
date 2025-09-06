# BeyondMimic: State-of-the-Art Humanoid Motion Tracking

BeyondMimic provides a powerful framework for creating highly dynamic and realistic humanoid motion, making sim-to-real deployment a reality.  ([Original Repo](https://github.com/HybridRobotics/whole_body_tracking))

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

*   **Sim-to-Real Ready:** Train dynamic motions from the LAFAN1 dataset, ready for real-world deployment.
*   **Guided Diffusion-Based Controllers:** Leverages state-of-the-art motion quality for enhanced control.
*   **WandB Integration:** Utilizes Weights & Biases (WandB) for streamlined motion management and registry.
*   **Modular Code Structure:** Organized code for easy navigation and extension.
*   **Flexible Motion Data:** Supports retargeted motions from various sources, including Unitree, KungfuBot, ASAP, and HuB.

## Overview

BeyondMimic is a versatile humanoid control framework that enables highly dynamic motion tracking. This repository focuses on motion tracking training, allowing you to generate sim-to-real-ready motions. For sim-to-sim and sim-to-real deployment, refer to the [motion\_tracking\_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

1.  **Install Isaac Lab:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for Isaac Lab v2.1.0. Conda installation is recommended.
2.  **Clone the Repository:** Clone this repository outside of your Isaac Lab directory:

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

4.  **Install the Library:** Using a Python interpreter with Isaac Lab installed:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

Leverage the WandB registry for automatic storage and loading of reference motions. Ensure the reference motion is retargeted and uses generalized coordinates.

1.  **Gather Reference Datasets:** Obtain motion datasets (follow original licenses) such as the Unitree-retargeted LAFAN1 Dataset, Sidekicks, Christiano Ronaldo celebration, and balance motions.
2.  **WandB Registry Setup:**
    *   Log in to your WandB account and access "Registry".
    *   Create a new registry collection named "Motions" with the artifact type "All Types".
3.  **Convert Motion Data:** Convert retargeted motions to include maximum coordinates information using forward kinematics:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This command uploads the processed motion file to the WandB registry.
4.  **Test Registry:** Replay the motion in Isaac Sim to confirm proper functionality:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

5.  **Debugging:**
    *   Set `WANDB_ENTITY` to your organization name.
    *   If `/tmp` is inaccessible, modify `csv_to_npz.py` (lines 319 & 326) to use a temporary folder of your choice.

### Policy Training

1.  **Train the Policy:**

    ```bash
    python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
    --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
    --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
    ```

### Policy Evaluation

1.  **Play the Trained Policy:**

    ```bash
    python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
    ```

    Locate the WandB run path in the run overview (format: `{your_organization}/{project_name}/{unique_id}`). Remember that `run_name` differs from `run_path`.

## Code Structure

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Core MDP functions.
    *   `commands.py`: Computes variables from reference motion, robot state, and error calculations.
    *   `rewards.py`: DeepMimic reward functions.
    *   `events.py`: Domain randomization terms.
    *   `observations.py`: Observation terms for motion tracking and data collection.
    *   `terminations.py`: Early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment (MDP) hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings.
*   **`scripts`**: Utility scripts for motion data preprocessing, policy training, and evaluation.