# BeyondMimic: Real-World Humanoid Motion Tracking with Guided Diffusion

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[//]: # (Link to Original Repo)
**[Original Repo](https://github.com/HybridRobotics/whole_body_tracking)**

BeyondMimic provides a robust framework for training and deploying highly dynamic and realistic humanoid motion tracking, achieving state-of-the-art performance in both simulated and real-world environments.

**Key Features:**

*   **Sim-to-Real Ready:** Train motion tracking models for real-world deployment.
*   **LAFAN1 Dataset Compatibility:** Train motion models using the LAFAN1 dataset without needing to tune parameters.
*   **Guided Diffusion Controllers:** Implement steerable test-time control with guided diffusion-based controllers for advanced motion control.
*   **WandB Registry Integration:** Seamlessly manage and load reference motions using the WandB registry.
*   **Modular Code Structure:**  Well-organized code with clear separation of concerns for easy customization and extension.

## Installation

This section outlines the steps to install and set up the BeyondMimic framework:

1.  **Install Isaac Lab:** Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), specifically the conda installation, for optimal use with Python scripts.
2.  **Clone the Repository:**
    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```
3.  **Fetch Robot Description Files:**
    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```
4.  **Install Python Library:**  Using a Python interpreter with Isaac Lab installed:
    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

This section guides you through setting up your reference motion datasets using the WandB registry.

*   **Data Sources:** The repository leverages datasets such as:
    *   Unitree-retargeted LAFAN1 Dataset ([HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
    *   Sidekicks ([KungfuBot](https://kungfu-bot.github.io/))
    *   Christiano Ronaldo celebration ([ASAP](https://github.com/LeCAR-Lab/ASAP))
    *   Balance motions ([HuB](https://hub-robot.github.io/))
*   **WandB Registry Configuration:**
    1.  Log in to your WandB account.
    2.  Access "Registry" under "Core" on the left-hand menu.
    3.  Create a new registry collection named "Motions" with the artifact type "All Types".
*   **Motion Conversion:** Convert retargeted motions to include maximum coordinates information (body pose, body velocity, and body acceleration):

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This will upload the processed motion file to the WandB registry.

*   **Testing the Registry:** Verify registry functionality by replaying the motion in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
*   **Debugging:**
    *   Ensure `WANDB_ENTITY` is set to your organization name.
    *   If `/tmp` is inaccessible, modify `csv_to_npz.py` (lines 319 & 326) to point to a temporary folder.

### Policy Training

Train your motion tracking policy with the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Evaluate your trained policy using:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The `wandb_path` can be found in your WandB run overview and follows the format `{your_organization}/{project_name}/` followed by a unique 8-character identifier.  Remember that `run_name` is distinct from `run_path`.

## Code Structure

The code is organized into the following key directories and files for modularity:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**:  Contains the core MDP (Markov Decision Process) components:
    *   `commands.py`: Functions for calculating variables from motion, robot state, and error.
    *   `rewards.py`: Implements reward functions.
    *   `events.py`: Includes domain randomization terms.
    *   `observations.py`: Observation terms for motion tracking.
    *   `terminations.py`: Early terminations and timeouts.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: MDP configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**:  PPO (Proximal Policy Optimization) hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings, including parameters.
*   **`scripts`**: Utility scripts for data preprocessing, policy training, and evaluation.

This structure promotes ease of use and customization.