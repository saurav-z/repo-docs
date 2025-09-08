# BeyondMimic: Train Realistic Humanoid Motion Tracking for Sim-to-Real Applications

BeyondMimic is a powerful framework enabling highly dynamic and realistic humanoid motion tracking, ready for deployment in both simulated and real-world environments.  Learn more and contribute at the original repository: [https://github.com/HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking).

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

*   **Sim-to-Real Ready Motion:** Train dynamic and realistic motion from the LAFAN1 dataset, out-of-the-box.
*   **Versatile Humanoid Control:** Provides highly dynamic motion tracking with state-of-the-art motion quality.
*   **Guided Diffusion-Based Control:** Includes steerable test-time control via guided diffusion-based controllers.
*   **WandB Registry Integration:** Leverages the WandB registry for efficient motion data management and loading.
*   **Modular Code Structure:** Well-organized code with clear separation of MDP functions, environment configurations, and robot settings.

## Overview

This repository provides the necessary code for training motion tracking models within the BeyondMimic framework. It focuses on the training process, enabling you to create sim-to-real-ready motions using the LAFAN1 dataset without the need for extensive parameter tuning.  For deployment in simulated or real-world environments, please see the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller) repository.

## Installation

To get started with BeyondMimic, follow these installation steps:

1.  **Install Isaac Lab:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for Isaac Lab v2.1.0.  Conda installation is recommended.
2.  **Clone the Repository:** Clone this repository outside of your Isaac Lab installation:

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```

3.  **Pull Robot Description Files:** Download and extract the robot description files:

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```

4.  **Install the Library:** Using a Python interpreter with Isaac Lab installed, install the library:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

This section guides you through the motion tracking process, including preprocessing, registry setup, policy training, and evaluation.

### Motion Preprocessing & Registry Setup

This project leverages WandB for managing a large number of motions.

1.  **Gather Datasets:** Obtain reference motion datasets (respecting their original licenses).  The repository supports data from:

    *   Unitree-retargeted LAFAN1 Dataset ([HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
    *   Sidekicks ([KungfuBot](https://kungfu-bot.github.io/))
    *   Christiano Ronaldo celebration ([ASAP](https://github.com/LeCAR-Lab/ASAP))
    *   Balance motions ([HuB](https://hub-robot.github.io/))

2.  **WandB Registry Setup:**

    *   Log in to your WandB account and create a new registry collection named "Motions" with artifact type "All Types".

3.  **Convert Motions:** Convert retargeted motions to include maximum coordinate information:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This will upload the processed motion files to the WandB registry.

4.  **Test Registry:** Verify the setup by replaying motions in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

5.  **Debugging:**
    *  Ensure `WANDB_ENTITY` is set to your organization name.
    *  If /tmp is inaccessible, adjust `csv_to_npz.py` (lines 319 & 326) to use an alternative temporary folder.

### Policy Training

Train your motion tracking policy using the following command:

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

The `wandb_path` can be found in your WandB run overview, using the format {your_organization}/{project_name}/ and an 8-character identifier.  Note that `run_name` differs from `run_path`.

## Code Structure

The following outlines the key components and directories within this repository:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`:** Defines the MDP for BeyondMimic, including:
    *   `commands.py`: Command library, error calculations, randomization, and adaptive sampling.
    *   `rewards.py`: DeepMimic reward functions and smoothing terms.
    *   `events.py`: Domain randomization terms.
    *   `observations.py`: Observation terms.
    *   `terminations.py`: Early terminations and timeouts.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`:**  Environment (MDP) hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`:** PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`:** Robot-specific settings, including parameters, joint calculations, and action scaling.
*   **`scripts`:** Utility scripts for motion data preprocessing, policy training, and evaluation.