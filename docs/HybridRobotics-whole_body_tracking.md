# BeyondMimic: Advanced Humanoid Motion Tracking with State-of-the-Art Quality

**BeyondMimic** is a cutting-edge framework empowering dynamic humanoid motion tracking, offering sim-to-real readiness and steerable control with guided diffusion-based controllers.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[**Original Repo**](https://github.com/HybridRobotics/whole_body_tracking) | [[Website]](https://beyondmimic.github.io/) | [[Arxiv]](https://arxiv.org/abs/2508.08241) | [[Video]](https://youtu.be/RS_MtKVIAzY)

## Key Features

*   **Sim-to-Real Ready Motion Tracking:** Train high-quality, sim-to-real-ready motion directly from the LAFAN1 dataset without parameter tuning.
*   **Dynamic Motion Quality:** Achieve state-of-the-art motion quality for realistic humanoid control.
*   **Guided Diffusion-Based Controllers:** Implement steerable test-time control for flexible motion generation.
*   **WandB Registry Integration:**  Leverage the WandB registry for streamlined motion management and automatic loading of reference motions.

## Installation

1.  **Install Isaac Lab:** Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), using conda installation is recommended.
2.  **Clone the Repository:** Clone this repository separately from your Isaac Lab installation.

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```

3.  **Pull Robot Description Files:** Download robot description files from Google Cloud Storage.

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```

4.  **Install the Library:**  Use a Python interpreter with Isaac Lab installed to install the library.

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking Workflow

### Motion Preprocessing & Registry Setup

1.  **Gather Datasets:** Download and prepare reference motion datasets, adhering to their original licenses. Example datasets include Unitree-retargeted LAFAN1, Sidekicks, Christiano Ronaldo celebration (ASAP), and Balance motions (HuB).  (Links to datasets in original README)
2.  **WandB Registry Configuration:**
    *   Log in to your WandB account and create a new registry collection named "Motions" with artifact type "All Types."
3.  **Convert Motions:** Use the following command to convert retargeted motions to include maximum coordinates information.

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This will automatically upload the processed motion file to the WandB registry.
4.  **Test Registry:** Verify the WandB registry by replaying motions in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
5.  **Debugging:**
    *   Ensure that `WANDB_ENTITY` is set to your organization name, not your personal username.
    *   If the `/tmp` folder is inaccessible, modify `csv_to_npz.py` (lines 319 & 326) to use an alternative temporary folder.

### Policy Training

Train your motion tracking policy:

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

*   The `wandb_path` can be found in your WandB run overview, formatted as `{your_organization}/{project_name}/{unique_identifier}`. Remember that `run_name` and `run_path` are distinct.

## Code Structure Overview

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Defines the Markov Decision Process (MDP) components:
    *   `commands.py`: Computes variables from reference motion, robot state, and error calculations.
    *   `rewards.py`: Implements reward functions and smoothing terms.
    *   `events.py`: Includes domain randomization terms.
    *   `observations.py`: Defines observation terms for motion tracking and data collection.
    *   `terminations.py`: Implements early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Contains environment hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: Defines PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**:  Contains robot-specific settings.
*   **`scripts`**: Contains utility scripts for motion data preprocessing, policy training, and evaluation.