# BeyondMimic: Dynamic Humanoid Motion Tracking for Sim-to-Real Deployment

**Achieve state-of-the-art humanoid motion tracking and seamless sim-to-real transfer with BeyondMimic, a versatile framework built for dynamic control.**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[**View the Original Repository**](https://github.com/HybridRobotics/whole_body_tracking) | [[Website]](https://beyondmimic.github.io/) | [[Arxiv]](https://arxiv.org/abs/2508.08241) | [[Video]](https://youtu.be/RS_MtKVIAzY)

## Key Features

*   **Sim-to-Real Ready:** Train motion tracking models for real-world deployment.
*   **No Parameter Tuning Required:** Achieve impressive motion quality on the LAFAN1 dataset without manual parameter adjustments.
*   **Guided Diffusion-Based Controllers:** Enables advanced control and test-time manipulation.
*   **WandB Registry Integration:** Seamlessly manage and load reference motions using the WandB registry.
*   **Modular Code Structure:** Well-organized code for easy understanding and development.

## Overview

BeyondMimic is a cutting-edge humanoid control framework designed for high-fidelity motion tracking. It provides exceptional dynamic motion quality, enabling real-world deployment and advanced test-time control. This repository focuses on the motion tracking training aspect, allowing you to train sim-to-real-ready motions using the LAFAN1 dataset with minimal configuration. For deployment details, refer to the [motion\_tracking\_controller](https://github.com/HybridRobotics/motion_tracking_controller) repository.

## Installation

### Prerequisites

*   [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) v2.1.0 (Conda installation recommended)

### Steps

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

### Motion Preprocessing & Registry Setup

Leverage the WandB registry to store and load reference motions efficiently.

1.  **Gather Datasets:** Ensure you have the required motion datasets (e.g., Unitree, Sidekicks, Christiano Ronaldo, Balance motions) and adhere to their respective licenses.
2.  **Create WandB Registry:**
    *   Log in to your WandB account and access the Registry under "Core."
    *   Create a new registry collection named "Motions" with artifact type "All Types."
3.  **Convert Motions:** Convert retargeted motions to include maximum coordinate information via forward kinematics.

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This command uploads the processed motion file to the WandB registry.

4.  **Test Registry:** Verify the functionality by replaying a motion in Isaac Sim.

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

    *   **Debugging:**
        *   Set `WANDB_ENTITY` to your organization name, not your personal username.
        *   If `/tmp` is inaccessible, modify `csv_to_npz.py` (lines 319 & 326) to use an alternate temporary directory.

### Policy Training

Train the motion tracking policy:

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

Find the `wandb_path` in the WandB run overview, formatted as `{your_organization}/{project_name}/` followed by an 8-character identifier. Remember, `run_name` and `run_path` are distinct.

## Code Structure

The code is organized for modularity and ease of use. Key directories and their roles are outlined below:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: MDP definition.
    *   `commands.py`: Computes variables from reference motion, robot state, and error calculations.
    *   `rewards.py`: Implements reward functions.
    *   `events.py`: Domain randomization terms.
    *   `observations.py`: Observation terms.
    *   `terminations.py`: Early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings.
*   **`scripts`**: Utility scripts for data preprocessing, training, and evaluation.