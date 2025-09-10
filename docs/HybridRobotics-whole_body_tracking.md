# BeyondMimic: Train Dynamic Humanoid Motion Tracking for Sim-to-Real Applications

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

**BeyondMimic is a cutting-edge framework for training realistic humanoid motion tracking policies, enabling seamless transfer from simulation to real-world robots.** This repository provides the code for training dynamic motion tracking models with state-of-the-art motion quality.

[**See the original repository**](https://github.com/HybridRobotics/whole_body_tracking)

**Key Features:**

*   **Sim-to-Real Ready:** Train motion tracking policies applicable to real-world deployments.
*   **LAFAN1 Dataset Compatibility:** Train any sim-to-real-ready motion within the LAFAN1 dataset without parameter tuning.
*   **WandB Registry Integration:** Leverages Weights & Biases (WandB) for efficient motion data management and experiment tracking.
*   **Modular Code Structure:** Organized code with clear documentation to facilitate development and customization.
*   **Guided Diffusion-Based Control:** Integrated with diffusion-based controllers for steerable test-time control (refer to the [motion\_tracking\_controller](https://github.com/HybridRobotics/motion_tracking_controller) for more details).

## Installation

1.  **Install Isaac Lab v2.1.0:**  Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Conda installation is recommended.
2.  **Clone the Repository:**  Clone this repository outside your `IsaacLab` directory.

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```
3.  **Get Robot Description Files:** Download and extract robot description files from GCS.

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```
4.  **Install the Library:** Install the library using a Python interpreter with Isaac Lab installed.

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking Workflow

### Motion Preprocessing & Registry Setup

This section describes how to prepare and upload motion data using the WandB registry.

*   **Gather Reference Motion Datasets:** Obtain reference motion datasets. The repository uses datasets like LAFAN1, Sidekicks (from KungfuBot), Christiano Ronaldo celebration (from ASAP), and balance motions (from HuB), according to their original licenses.
*   **WandB Registry Setup:** Create a new registry collection in your WandB account, naming it "Motions" with the artifact type "All Types".
*   **Convert Motions:** Convert retargeted motion data to include maximum coordinate information using forward kinematics.

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This command automatically uploads the processed motion files to the WandB registry.
*   **Test Registry:** Verify proper functionality by replaying motions within Isaac Sim.

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
*   **Debugging:**
    *   Set `WANDB_ENTITY` to your organization name in the environment variables.
    *   If `/tmp` is inaccessible, modify `csv_to_npz.py` (lines 319 & 326) to use a temporary directory of your choosing.

### Policy Training

Train your motion tracking policy with the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Evaluate the trained policy with:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

Obtain the WandB run path from your run overview. The format is: `{your_organization}/{project_name}/` along with an 8-character identifier. Note: `run_name` is distinct from `run_path`.

## Code Structure Overview

The code is modular and easy to navigate.  Key directories and files include:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**:
    *   `commands.py`: Calculates variables, including error calculations.
    *   `rewards.py`: Defines reward functions.
    *   `events.py`: Implements domain randomization.
    *   `observations.py`: Defines observation terms.
    *   `terminations.py`: Implements early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Defines environment configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: Contains PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Contains robot-specific settings and calculations.
*   **`scripts`**: Includes utility scripts for data preprocessing, policy training, and evaluation.