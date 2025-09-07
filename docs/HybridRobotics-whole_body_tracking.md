# BeyondMimic: Advanced Humanoid Motion Tracking for Sim-to-Real Applications

**BeyondMimic** enables cutting-edge humanoid motion tracking with state-of-the-art motion quality, empowering dynamic control and seamless sim-to-real deployment.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[**[Website]**](https://beyondmimic.github.io/) | [**[Arxiv]**](https://arxiv.org/abs/2508.08241) | [**[Video]**](https://youtu.be/RS_MtKVIAzY) | [**[Original Repo]**](https://github.com/HybridRobotics/whole_body_tracking)

## Key Features

*   **Train Sim-to-Real Ready Motion:** Train high-fidelity motion tracking models for the LAFAN1 dataset without parameter tuning.
*   **Dynamic Motion Quality:** Achieve state-of-the-art motion quality.
*   **Versatile Control Framework:**  Provides a framework for dynamic motion tracking and guided diffusion-based control.
*   **WandB Integration:** Leverage Weights & Biases for motion registry and policy training.

## Installation

**Prerequisites:**

*   Isaac Lab v2.1.0 ([Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)) (Recommended: Conda installation).
*   Python 3.10

**Steps:**

1.  **Clone the Repository:** Clone the repository outside the Isaac Lab directory.

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```
2.  **Fetch Robot Description Files:**

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```
3.  **Install the Library:**  Install the library using a Python interpreter with Isaac Lab installed.

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking Workflow

### 1. Motion Preprocessing & Registry Setup

*   **Gather Reference Motion Datasets:** Obtain datasets and follow the original licenses.  Supported datasets include: LAFAN1 (retargeted), Sidekicks, Christiano Ronaldo celebration (ASAP), and Balance motions (HuB).
*   **WandB Registry:**
    *   Log in to your WandB account.
    *   Create a new registry collection named "Motions" with artifact type "All Types".
*   **Convert Motions:** Convert retargeted motions to include maximum coordinates information.

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This uploads the processed motion to the WandB registry.
*   **Test Registry:** Verify registry functionality by replaying motions in Isaac Sim.

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
*   **Debugging:**
    *   Ensure `WANDB_ENTITY` is set to your organization name.
    *   If `/tmp` is inaccessible, adjust `csv_to_npz.py` (lines 319 & 326) to use an alternate temporary folder.

### 2. Policy Training

*   Train a policy using the following command:

    ```bash
    python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
    --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
    --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
    ```

### 3. Policy Evaluation

*   Play a trained policy:

    ```bash
    python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
    ```

    Find the WandB run path in the run overview, which follows the format `{your_organization}/{project_name}/{unique_identifier}`.

## Code Structure Overview

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**:  Core MDP functionality.
    *   `commands.py`: Computes variables and errors.
    *   `rewards.py`: Defines reward functions.
    *   `events.py`: Handles domain randomization.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`: Defines early terminations.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment (MDP) configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**:  PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings (armature parameters, joint calculations, action scales).
*   **`scripts`**:  Utility scripts for data processing, policy training, and evaluation.