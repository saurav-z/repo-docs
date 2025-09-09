# BeyondMimic: State-of-the-Art Humanoid Motion Tracking

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[**[Original Repository]**](https://github.com/HybridRobotics/whole_body_tracking) | [[Website]](https://beyondmimic.github.io/) | [[Arxiv]](https://arxiv.org/abs/2508.08241) | [[Video]](https://youtu.be/RS_MtKVIAzY)

BeyondMimic provides a versatile humanoid control framework to achieve high-quality, dynamic motion tracking, ready for real-world deployment and steerable test-time control. This repository contains the code for training motion tracking models.

**Key Features:**

*   **Sim-to-Real Ready:** Train motion tracking models that can be directly deployed in the real world.
*   **LAFAN1 Dataset Compatibility:** Train sim-to-real-ready motion without parameter tuning.
*   **WandB Registry Integration:** Seamlessly manage and load reference motions using the WandB registry.
*   **Modular Code Structure:** Clearly organized code for easy understanding and extension.
*   **Guided Diffusion-Based Controllers:** Utilize cutting-edge controllers for advanced motion control.

## Installation

1.  **Install Isaac Lab:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for Isaac Lab v2.1.0 (Conda recommended).

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

4.  **Install the Library:**

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking Workflow

### Motion Preprocessing & Registry Setup

To efficiently handle and utilize reference motions, the WandB registry is employed.

1.  **Gather Datasets:** Ensure you have the reference motion datasets (e.g., LAFAN1, Sidekicks, Christiano Ronaldo celebration, Balance motions).  Follow original licenses.

    *   LAFAN1 Dataset is available on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
2.  **WandB Registry Setup:**
    *   Log in to your WandB account.
    *   Create a new registry collection in WandB with the name "Motions" and artifact type "All Types".

3.  **Convert Motions:** Convert the retargeted motions to include maximum coordinates information.

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This uploads the processed motion file to the WandB registry.

4.  **Test Registry:** Verify the WandB registry functionality by replaying motions in Isaac Sim.

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```

5.  **Debugging**
    *   Make sure to export WANDB_ENTITY to your organization name, not your personal username.
    *   If /tmp folder is not accessible, modify csv_to_npz.py L319 & L326 to a temporary folder of your choice.

### Policy Training

1.  **Train Policy:**

    ```bash
    python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
    --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
    --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
    ```

### Policy Evaluation

1.  **Play Trained Policy:**

    ```bash
    python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
    ```

    The WandB run path follows the format `{your_organization}/{project_name}/` along with a unique 8-character identifier. Note that `run_name` is different from `run_path`.

## Code Structure Overview

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**:  Defines the MDP for BeyondMimic.
    *   `commands.py`: Computes variables from reference motions, robot state, and error calculations.
    *   `rewards.py`:  Implements reward functions.
    *   `events.py`: Implements domain randomization.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`: Implements early terminations.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment (MDP) configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings (armature parameters, joint calculations).
*   **`scripts`**: Utility scripts for data preprocessing, training, and evaluation.