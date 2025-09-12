# BeyondMimic: Advanced Humanoid Motion Tracking for Realistic Simulations

**BeyondMimic offers a cutting-edge framework for dynamic humanoid motion tracking, providing high-quality motion replication in both simulated and real-world environments.** ([Original Repository](https://github.com/HybridRobotics/whole_body_tracking))

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

*   **Sim-to-Real Ready:** Train motion tracking models suitable for real-world robot deployment.
*   **State-of-the-Art Motion Quality:** Achieve highly dynamic and realistic motion replication.
*   **Guided Diffusion-Based Controllers:** Leverage steerable test-time control for advanced manipulation.
*   **LAFAN1 Dataset Compatibility:** Train motion tracking models on the widely-used LAFAN1 dataset without parameter tuning.
*   **WandB Integration:** Utilizes WandB for efficient motion registry and experiment tracking.

## Overview

BeyondMimic is a versatile humanoid control framework designed for advanced motion tracking. This repository specifically focuses on the motion tracking training aspects, allowing you to create sim-to-real-ready motions. For deployment, please refer to the [motion\_tracking\_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

### Prerequisites

*   Isaac Lab v2.1.0 ([Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)) - Conda installation is recommended.
*   Python 3.10

### Steps

1.  **Clone the Repository:**

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

3.  **Install the Library:**

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

*   **Reference Datasets:** Gather motion data from sources like LAFAN1 (retargeted version available on HuggingFace), Sidekicks (KungfuBot), Christiano Ronaldo celebration (ASAP), and Balance motions (HuB)
*   **WandB Registry Setup:** Create a new registry collection in WandB named "Motions" with artifact type "All Types".
*   **Motion Conversion:** Convert retargeted motions to include the maximum coordinates using the following command:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This uploads processed motion files to the WandB registry.

*   **Testing:** Verify registry functionality by replaying motions in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
*   **Debugging:**
    *   Set the `WANDB_ENTITY` environment variable to your organization's name.
    *   If the `/tmp` folder is inaccessible, modify `csv_to_npz.py` at lines 319 & 326 to use a different temporary folder.

### Policy Training

Train policies using the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Evaluate trained policies with:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

Find the `wandb_path` in your WandB run overview (e.g., `{your_organization}/{project_name}/` plus an 8-character ID). Note that `run_name` is distinct from `run_path`.

## Code Structure

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Defines the MDP for BeyondMimic, including:
    *   `commands.py`: Computes variables from reference motion, robot state, and errors.
    *   `rewards.py`: Implements reward functions and smoothing terms.
    *   `events.py`: Implements domain randomization.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`: Defines early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Contains environment (MDP) hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: Contains PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Contains robot-specific settings.
*   **`scripts`**: Utility scripts for motion preprocessing, training, and evaluation.

---

**[Back to Top](#)**