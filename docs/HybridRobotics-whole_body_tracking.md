# BeyondMimic: Whole-Body Motion Tracking for Dynamic Humanoid Control

**BeyondMimic offers a cutting-edge framework for robust whole-body motion tracking, enabling dynamic and high-quality humanoid control for both real-world deployment and simulation.** ([Original Repository](https://github.com/HybridRobotics/whole_body_tracking))

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

*   **Sim-to-Real Ready:** Train dynamic motion tracking models ready for real-world deployment.
*   **High-Quality Motion:** Achieve state-of-the-art motion quality.
*   **Guided Diffusion Control:** Offers steerable test-time control based on guided diffusion.
*   **LAFAN1 Compatibility:** Train motion using the LAFAN1 dataset without extensive parameter tuning.
*   **WandB Registry Integration:** Utilize WandB for streamlined motion management and data tracking.
*   **Modular Code Structure:** Facilitates ease of navigation and customization for developers.

## Installation

1.  **Install Isaac Lab:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for Isaac Lab v2.1.0, preferably using Conda.
2.  **Clone the Repository:** Clone this repository separately from the Isaac Lab installation:

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

4.  **Install the Library:**  Using a Python interpreter with Isaac Lab installed:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

This section details the steps required for setting up motion data using the WandB registry.

*   **Gather Datasets:**
    *   Unitree-retargeted LAFAN1 Dataset ([HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
    *   Sidekicks ([KungfuBot](https://kungfu-bot.github.io/))
    *   Christiano Ronaldo celebration ([ASAP](https://github.com/LeCAR-Lab/ASAP))
    *   Balance motions ([HuB](https://hub-robot.github.io/))
*   **WandB Registry Setup:**
    1.  Log in to your WandB account.
    2.  Create a new registry collection in WandB ("Motions", artifact type "All Types").
*   **Convert Motions:** Convert retargeted motions to include maximum coordinates:

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This will automatically upload the processed motion file to the WandB registry.
*   **Test Registry:** Verify the setup by replaying the motion in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
*   **Debugging:**
    *   Ensure `WANDB_ENTITY` is set to your organization name, not your username.
    *   If /tmp is inaccessible, change the temporary folder locations in `csv_to_npz.py` (lines 319 & 326).

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

    The WandB run path is located in the run overview (e.g., `{your_organization}/{project_name}/` followed by an 8-character identifier).

## Code Structure

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Core functions for the MDP.
    *   `commands.py`: Computes variables from reference motions, robot states, and error calculations.
    *   `rewards.py`:  Implements reward functions.
    *   `events.py`: Implements domain randomization.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`: Implements early terminations.

*   `source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`:  Environment (MDP) hyperparameter configuration.

*   `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`: PPO hyperparameters.

*   `source/whole_body_tracking/whole_body_tracking/robots`: Robot-specific settings.

*   `scripts`: Utility scripts for data preprocessing, policy training, and evaluation.