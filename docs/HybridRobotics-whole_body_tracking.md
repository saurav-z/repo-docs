# BeyondMimic: Cutting-Edge Whole-Body Motion Tracking

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[//]: # (Link back to original repo - placed at the top for better SEO)
[GitHub Repository](https://github.com/HybridRobotics/whole_body_tracking)

BeyondMimic empowers dynamic humanoid motion tracking with state-of-the-art quality for real-world deployment.  This repository provides the code for training motion tracking models.

**Key Features:**

*   **Sim-to-Real Ready:** Train motion in the LAFAN1 dataset and deploy it to real-world scenarios.
*   **WandB Integration:** Leverages Weights & Biases for managing reference motions and experiment tracking.
*   **Versatile Control:** Enables test-time control with guided diffusion-based controllers.
*   **Modular Code Structure:** Well-organized code for ease of use and customization.
*   **Ready-to-Use:** Includes scripts for motion preprocessing, policy training, and evaluation.

## Overview

This repository focuses on the motion tracking training aspect of the BeyondMimic framework. It leverages the LAFAN1 dataset and allows you to train motion for real-world scenarios. For deployment, please refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

Follow these steps to get started:

1.  **Install Isaac Lab v2.1.0:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Conda installation is recommended.
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

4.  **Install the Library:** Use a Python interpreter with Isaac Lab installed:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

This project uses WandB for managing a large set of motions automatically.

1.  **Gather Reference Motion Datasets:** Download the required datasets, following their respective licenses. Datasets include:
    *   Unitree-retargeted LAFAN1 Dataset (HuggingFace: [https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
    *   Sidekicks (KungfuBot: [https://kungfu-bot.github.io/](https://kungfu-bot.github.io/))
    *   Christiano Ronaldo celebration (ASAP: [https://github.com/LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP))
    *   Balance motions (HuB: [https://hub-robot.github.io/](https://hub-robot.github.io/))
2.  **WandB Registry Setup:**
    *   Log in to your WandB account and create a new registry collection named "Motions" with the artifact type "All Types".
3.  **Convert Motions:** Convert the retargeted motions to include maximum coordinates information.

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This will upload the processed motion file to the WandB registry.
4.  **Test the Registry:** Verify that the registry works by replaying the motion in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
5.  **Debugging:**
    *   Set `WANDB_ENTITY` to your organization name.
    *   If you can't access the `/tmp` folder, change the temporary folder in `csv_to_npz.py` (lines 319 & 326).

### Policy Training

Train your policy using the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Play the trained policy:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

Find the WandB run path in the run overview (e.g., `{your_organization}/{project_name}/{unique_identifier}`). Remember that `run_name` is different from `run_path`.

## Code Structure

This section provides an overview of the code organization:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Contains the core functions to define the Markov Decision Process (MDP).
    *   **`commands.py`**: Computes variables from reference motion, robot state, and error calculations.
    *   **`rewards.py`**: Implements the DeepMimic reward functions.
    *   **`events.py`**: Implements domain randomization terms.
    *   **`observations.py`**: Implements observation terms.
    *   **`terminations.py`**: Implements early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Contains the environment (MDP) hyperparameters configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: Contains the PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale calculation.
*   **`scripts`**: Includes utility scripts for motion preprocessing, training policies, and evaluating trained policies.