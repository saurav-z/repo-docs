# BeyondMimic: High-Fidelity Humanoid Motion Tracking 

**Achieve state-of-the-art dynamic motion tracking and sim-to-real readiness for humanoid robots with BeyondMimic, built on Isaac Sim and Isaac Lab.**

*   **Link to Original Repository:** [https://github.com/HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)

## Key Features

*   **Sim-to-Real Ready:** Train motion tracking models that can be deployed on real-world humanoid robots.
*   **LAFAN1 Dataset Compatibility:** Train motion tracking models on the LAFAN1 dataset without requiring parameter tuning.
*   **WandB Registry Integration:** Streamlines motion data management and loading.
*   **Guided Diffusion-Based Controllers:** Supports steerable test-time control.
*   **Modular Code Structure:** Well-organized code for easy navigation and customization.
*   **Support for Multiple Datasets:**  Works with Unitree, Sidekicks, Christiano Ronaldo celebration, and HuB datasets.

## Overview

BeyondMimic is a comprehensive humanoid control framework developed to provide exceptional dynamic motion tracking capabilities. It is specifically designed for real-world deployment, offering the highest motion quality, and incorporates guided diffusion-based controllers for flexible control at test time. This repository focuses on the motion tracking training aspects of BeyondMimic, enabling the training of sim-to-real-ready motions, specifically within the LAFAN1 dataset, requiring no parameter tuning.

For information on sim-to-sim and sim-to-real deployment, please refer to the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

### Prerequisites

*   **Isaac Lab:** Install Isaac Lab v2.1.0.  It is recommended to use the conda installation to facilitate calling Python scripts from the terminal. The installation guide is available [here](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
*   **Python 3.10**: Required for project compatibility.
*   **Platform:** Linux-64 is the supported platform.

### Steps

1.  **Clone the Repository:**
    Clone this repository separately from your Isaac Lab installation:

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
    Use a Python interpreter with Isaac Lab installed to install the library:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

Leveraging the WandB registry, this section outlines how to manage and load reference motions:

*   **Dataset Preparation:** Gather reference motion datasets (ensure compliance with original licenses), following conventions as .csv of Unitree's dataset. Datasets such as Unitree-retargeted LAFAN1 Dataset, Sidekicks, Christiano Ronaldo celebration (from ASAP), and Balance motions (from HuB) are supported.
*   **WandB Registry Configuration:** Log in to your WandB account and create a new registry collection named "Motions" with the artifact type "All Types."
*   **Motion Conversion:** Convert retargeted motions to include maximum coordinate information (body pose, velocity, acceleration) using forward kinematics:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This step automatically uploads the processed motion file to the WandB registry.
*   **Registry Verification:** Test the WandB registry setup by replaying the motion in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

*   **Debugging Tips:**
    *   Ensure `WANDB_ENTITY` is set to your organization's name, not your username.
    *   If the `/tmp` folder is inaccessible, adjust `csv_to_npz.py` lines 319 and 326 to use a different temporary folder.

### Policy Training

Train your policy with the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

To evaluate a trained policy, use the command below:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The WandB run path, available in the run overview, follows the format `{your_organization}/{project_name}/` combined with a unique 8-character identifier. Note that `run_name` is distinct from `run_path`.

## Code Structure

The following provides an overview of the project's code structure for easier navigation and modification:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: This directory defines the MDP for BeyondMimic, with these key functions:

    *   `commands.py`: Computes variables from the reference motion, robot state, and error calculations. Includes pose and velocity error computations, state randomization, and adaptive sampling.
    *   `rewards.py`: Implements DeepMimic reward functions.
    *   `events.py`: Implements domain randomization terms.
    *   `observations.py`: Implements observation terms for motion tracking and data collection.
    *   `terminations.py`: Implements early terminations and timeouts.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Contains environment (MDP) hyperparameters for the tracking task.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: Includes PPO hyperparameters for the tracking task.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Contains robot-specific settings such as armature parameters, joint stiffness/damping calculations, and action scaling.
*   **`scripts`**: Includes utility scripts for preprocessing motion data, and training/evaluating policies.