# BeyondMimic: State-of-the-Art Humanoid Motion Tracking

BeyondMimic revolutionizes humanoid control with dynamic motion tracking and guided diffusion-based controllers, enabling impressive real-world performance. ([Original Repo](https://github.com/HybridRobotics/whole_body_tracking))

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

*   **High-Fidelity Motion Tracking:** Achieve state-of-the-art motion quality, directly deployable in the real world.
*   **Sim-to-Real Ready:** Train sim-to-real-ready motion using the LAFAN1 dataset without extensive parameter tuning.
*   **Guided Diffusion-Based Controllers:** Utilize advanced control methods for test-time control and dynamic behaviors.
*   **WandB Integration:** Seamlessly integrate with Weights & Biases for motion data management and logging.
*   **Modular Code Structure:** Easy to navigate code to enable developers to expand the project.

## Overview

This repository provides the code for training motion tracking in BeyondMimic, a versatile framework for humanoid control.  It enables the creation of dynamic and lifelike motions for simulated and real-world humanoid robots. You can train sim-to-real-ready motions within the LAFAN1 dataset with minimal parameter tuning.

For details on sim-to-sim and sim-to-real deployment, consult the  [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

1.  **Install Isaac Lab:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for Isaac Lab v2.1.0, using conda installation for ease of use.
2.  **Clone Repository:** Clone this repository outside of your `IsaacLab` directory:

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```

3.  **Get Robot Description Files:** Download robot description files from GCS:

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```

4.  **Install Library:** Install the library using a Python interpreter that has Isaac Lab installed:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

Leverage the WandB registry to store and load reference motions automatically. Ensure reference motions are retargeted and utilize generalized coordinates.

*   **Gather Datasets:**  Acquire reference motion datasets.  Example datasets:

    *   Unitree-retargeted LAFAN1 Dataset ([HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
    *   Sidekicks ([KungfuBot](https://kungfu-bot.github.io/))
    *   Christiano Ronaldo celebration ([ASAP](https://github.com/LeCAR-Lab/ASAP))
    *   Balance motions ([HuB](https://hub-robot.github.io/))

*   **WandB Registry Setup:**

    1.  Log in to your WandB account.
    2.  Navigate to "Registry" under "Core."
    3.  Create a new registry collection named "Motions" with the artifact type "All Types."

*   **Convert Motions:** Transform retargeted motions using forward kinematics for maximum coordinate information (body pose, velocity, and acceleration):

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This will upload the processed file to the WandB registry.

*   **Test with Isaac Sim:** Verify the registry by replaying the motion in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

*   **Debugging Tips:**

    *   Set `WANDB_ENTITY` to your organization name.
    *   Modify `csv_to_npz.py` (lines 319 & 326) to use a temporary folder if your `/tmp` directory is inaccessible.

### Policy Training

*   Train a policy using the following command:

    ```bash
    python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
    --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
    --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
    ```

### Policy Evaluation

*   Evaluate the trained policy:

    ```bash
    python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
    ```

    Find the WandB run path in the run overview (format: `{your_organization}/{project_name}/` followed by an 8-character identifier).  Note: `run_name` is different from `run_path`.

## Code Structure

The code is organized into modular components, designed for clear understanding and expansion:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`:** Contains core MDP functions.

    *   `commands.py`:  Calculates variables from reference motions, robot states, and errors.
    *   `rewards.py`:  Implements DeepMimic reward functions and smoothing.
    *   `events.py`:  Provides domain randomization elements.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`:  Defines early termination and timeout conditions.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`:** Environment (MDP) hyperparameter configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`:** PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`:** Robot-specific settings (armature parameters, joint stiffness/damping, action scaling).
*   **`scripts`:**  Utility scripts (motion preprocessing, policy training and evaluation).