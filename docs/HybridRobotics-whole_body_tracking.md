# BeyondMimic: Train Dynamic Humanoid Motion Tracking with Ease

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[**[Website]**](https://beyondmimic.github.io/)
[**[Arxiv]**](https://arxiv.org/abs/2508.08241)
[**[Video]**](https://youtu.be/RS_MtKVIAzY)

This repository provides the code for BeyondMimic, a versatile humanoid control framework that excels at dynamic motion tracking, making it easy to train sim-to-real-ready motions without parameter tuning.  [**View the original repository here**](https://github.com/HybridRobotics/whole_body_tracking).

## Key Features

*   **Sim-to-Real Ready:** Train motion tracking models directly applicable to real-world deployments.
*   **No Parameter Tuning Required:** Train any motion in the LAFAN1 dataset with minimal setup.
*   **WandB Integration:** Leverage Weights & Biases (WandB) for streamlined motion management and experiment tracking.
*   **Modular Code Structure:** Well-organized code for easy navigation and expansion.
*   **Reproducible Results:** Training and evaluation scripts for consistent and reliable results.

## Installation

Follow these steps to set up the environment:

1.  **Install Isaac Lab:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for Isaac Lab v2.1.0. Conda is recommended.

2.  **Clone the Repository:** Clone this repository separately from the Isaac Lab installation:

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```

3.  **Get Robot Description Files:** Pull robot description files from Google Cloud Storage (GCS):

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```

4.  **Install the Library:** Install the Python library using a Python interpreter with Isaac Lab installed:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking Workflow

This section guides you through the motion tracking process:

### Motion Preprocessing & Registry Setup

*   **Reference Motion Datasets:** Gather reference motion datasets.  The code supports various datasets including Unitree-retargeted LAFAN1 (available on HuggingFace), Sidekicks (from KungfuBot), Christiano Ronaldo celebrations (from ASAP), and Balance motions (from HuB). Please adhere to the original licenses of these datasets.
*   **WandB Registry:**
    1.  Log in to your WandB account.
    2.  Access "Registry" under "Core" on the left.
    3.  Create a new registry collection named "Motions" with the artifact type "All Types".
*   **Convert Motions:**  Convert retargeted motions to include maximum coordinate information (body pose, velocity, and acceleration):

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This will automatically upload the processed motion to the WandB registry.

*   **Test Registry:** Verify the WandB registry functionality by replaying the motion in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

*   **Debugging:**
    *   Ensure `WANDB_ENTITY` is set to your organization name.
    *   If `/tmp` is inaccessible, modify `csv_to_npz.py` (lines 319 and 326) to use an alternative temporary folder.

### Policy Training

Train your policy using the following command:

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

The `wandb_path` can be found in the WandB run overview and is in the format of {your_organization}/{project_name}/ along with a unique 8-character identifier.  Note the distinction between `run_name` and `run_path`.

## Code Structure Overview

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Defines the core MDP components.
    *   `commands.py`: Computes variables from reference motion, robot state, and errors.
    *   `rewards.py`: Implements DeepMimic reward functions and smoothing terms.
    *   `events.py`: Implements domain randomization.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`: Handles early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**:  Environment configuration (MDP hyperparameters).
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**:  PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings (armature, joint properties, action scaling).
*   **`scripts`**: Utility scripts for motion preprocessing, training, and evaluation.