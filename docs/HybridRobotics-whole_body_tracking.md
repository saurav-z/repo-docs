# BeyondMimic: Advanced Humanoid Motion Tracking with Sim-to-Real Capabilities

**Unlock realistic and dynamic humanoid motion tracking with BeyondMimic, a versatile framework leveraging state-of-the-art motion quality and guided diffusion-based controllers.**  ([Original Repository](https://github.com/HybridRobotics/whole_body_tracking))

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[Website](https://beyondmimic.github.io/) | [Arxiv](https://arxiv.org/abs/2508.08241) | [Video](https://youtu.be/RS_MtKVIAzY)

## Key Features

*   **Sim-to-Real Ready Motion:** Train high-quality motion tracking models that can be deployed in real-world scenarios.
*   **LAFAN1 Dataset Compatibility:**  Train motion tracking models using the LAFAN1 dataset without the need for parameter tuning.
*   **WandB Registry Integration:** Seamlessly manage and load reference motions using the Weights & Biases (WandB) registry.
*   **Guided Diffusion-Based Controllers:** Utilize steerable test-time control for advanced motion control.
*   **Modular Code Structure:**  Organized codebase for easy navigation, modification, and extension.

## Overview

BeyondMimic is a cutting-edge humanoid control framework designed for creating highly dynamic motion tracking with exceptional quality, suitable for both real-world deployment and advanced control.  This repository provides the necessary code for training motion tracking models. For deployment and controller details, refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

1.  **Install Isaac Lab:** Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), preferably using conda.
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

## Motion Tracking

### Motion Preprocessing & Registry Setup

BeyondMimic utilizes a WandB registry for efficient motion management:

1.  **Gather Datasets:** Obtain reference motion datasets.  The following are supported:
    *   Unitree-retargeted LAFAN1 Dataset ([HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
    *   Sidekicks ([KungfuBot](https://kungfu-bot.github.io/))
    *   Christiano Ronaldo celebration ([ASAP](https://github.com/LeCAR-Lab/ASAP))
    *   Balance motions ([HuB](https://hub-robot.github.io/))

2.  **WandB Registry Setup:**
    *   Log in to your WandB account.
    *   Create a new registry collection named "Motions" with artifact type "All Types".

3.  **Convert and Upload Motion Data:** Convert retargeted motions into a format that includes the maximum coordinates information (body pose, velocity, and acceleration):

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This automatically uploads the processed motion file to the WandB registry.

4.  **Test Motion Replay:** Verify registry functionality within Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

5.  **Debugging:**
    *   Ensure `WANDB_ENTITY` is set to your organization name.
    *   Modify `csv_to_npz.py` (lines 319 & 326) to specify a temporary directory if needed.

### Policy Training

Train a policy using:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Evaluate a trained policy using:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The `wandb_path` can be found in the WandB run overview.

## Code Structure

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Contains core MDP functions, including:
    *   `commands.py`: Computes variables from motion, robot state, and error.
    *   `rewards.py`: Implements reward functions.
    *   `events.py`: Domain randomization.
    *   `observations.py`: Observation terms.
    *   `terminations.py`: Early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment (MDP) configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings.
*   **`scripts`**: Utility scripts for data preprocessing, training, and evaluation.