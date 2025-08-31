# BeyondMimic: Train and Deploy Dynamic Humanoid Motion Tracking with State-of-the-Art Quality

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[**Original Repository**](https://github.com/HybridRobotics/whole_body_tracking) | [[Website]](https://beyondmimic.github.io/) | [[Arxiv]](https://arxiv.org/abs/2508.08241) | [[Video]](https://youtu.be/RS_MtKVIAzY)

BeyondMimic offers a powerful framework for training and deploying highly dynamic and realistic humanoid motion tracking, enabling sim-to-real readiness with unparalleled motion quality.

## Key Features

*   **Sim-to-Real Ready Training:** Train motion tracking models directly from the LAFAN1 dataset, requiring minimal parameter tuning.
*   **State-of-the-Art Motion Quality:** Achieve highly dynamic motion tracking with superior realism.
*   **Guided Diffusion-Based Controllers:** Leverage steerable test-time control for flexible motion adaptation.
*   **WandB Registry Integration:** Seamlessly manage and load reference motions using the Weights & Biases registry.
*   **Modular Code Structure:** Navigate and extend the project easily with a well-defined code organization.

## Installation

1.  **Install Isaac Lab:** Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), preferably using conda.
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
4.  **Install the Library:**  Using a Python interpreter with Isaac Lab installed:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

1.  **Gather Reference Motion Datasets:** Obtain datasets following their respective licenses.  Examples include:

    *   Unitree-retargeted LAFAN1 Dataset (HuggingFace):  [https://huggingface.co/datasets/lvhaidong/LAFAN1\_Retargeting\_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    *   Sidekicks (KungfuBot): [https://kungfu-bot.github.io/](https://kungfu-bot.github.io/)
    *   Christiano Ronaldo celebration (ASAP): [https://github.com/LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP)
    *   Balance motions (HuB): [https://hub-robot.github.io/](https://hub-robot.github.io/)
2.  **WandB Registry Setup:**
    *   Log in to your WandB account.
    *   In the "Core" section on the left, go to "Registry" and create a new registry collection named "Motions" with the artifact type "All Types".
3.  **Convert Motions:**  Transform retargeted motions to include maximum coordinate information (body pose, velocity, and acceleration):

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This will automatically upload the processed motion files to the WandB registry.
4.  **Test Registry:** Replay a motion in Isaac Sim to ensure the WandB registry is functioning correctly:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
5.  **Debugging:**
    *   Ensure `WANDB_ENTITY` is set to your organization name.
    *   If `/tmp` is inaccessible, modify `csv_to_npz.py` lines 319 & 326 to use a temporary directory of your choice.

### Policy Training

Train a policy with the following command:

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

Find the `wandb_path` in the run overview; it follows this format: `{your_organization}/{project_name}/` along with a unique 8-character identifier.

## Code Structure

The code is structured for modularity and maintainability:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`:** Contains the core MDP functions:

    *   `commands.py`: Computes variables from reference motion, robot state, and error calculations.
    *   `rewards.py`: Implements reward functions.
    *   `events.py`:  Implements domain randomization.
    *   `observations.py`: Defines observation terms.
    *   `terminations.py`:  Handles early terminations and timeouts.

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`:**  Environment configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`:** PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`:** Robot-specific settings.
*   **`scripts`:** Utility scripts for data preprocessing, training, and evaluation.