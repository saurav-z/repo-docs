# BeyondMimic: Dynamic Humanoid Motion Tracking with Guided Diffusion

**BeyondMimic** offers a cutting-edge framework for realistic and dynamic humanoid motion tracking, providing state-of-the-art motion quality and test-time control using guided diffusion controllers.  [See the original repo](https://github.com/HybridRobotics/whole_body_tracking) for more details.

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

*   **High-Fidelity Motion Tracking:** Achieve state-of-the-art motion quality for realistic humanoid movements.
*   **Sim-to-Real Ready:** Train motion tracking models on the LAFAN1 dataset with adaptive sampling, minimizing the need for parameter tuning.
*   **Guided Diffusion Controllers:** Leverage diffusion-based controllers for advanced test-time control.
*   **WandB Integration:** Utilize Weights & Biases (WandB) for streamlined motion management and experiment tracking.
*   **Modular Code Structure:** A well-organized code base for easy navigation and extension.

## Installation

This section outlines the steps to set up the BeyondMimic environment.

1.  **Install Isaac Lab:** Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Conda installation is recommended.
2.  **Clone the Repository:**

    ```bash
    # Option 1: SSH
    git clone git@github.com:HybridRobotics/whole_body_tracking.git

    # Option 2: HTTPS
    git clone https://github.com/HybridRobotics/whole_body_tracking.git
    ```
3.  **Pull Robot Description Files:**

    ```bash
    cd whole_body_tracking
    curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz
    ```
4.  **Install the Library:** Activate your Python environment with Isaac Lab installed and then run:

    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking Workflow

This section describes the key steps involved in training and evaluating motion tracking policies.

### Motion Preprocessing and Registry Setup

The following steps will get you started using the WandB registry:

1.  **Gather Datasets:** Obtain the necessary motion datasets. The repository supports various datasets.
2.  **WandB Registry Setup:** Create a new registry collection named "Motions" with artifact type "All Types" in your WandB account.
3.  **Convert Motions:** Utilize the provided script to convert motion data to a suitable format.

    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```

    This step will automatically upload the processed motion file to the WandB registry.
4.  **Test Registry:** Verify that the registry is working correctly by replaying a motion in Isaac Sim:

    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
5.  **Debugging:**  Refer to the original README for debugging tips, including environment variables and temporary folder modifications.

### Policy Training

Train your motion tracking policies using the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

Evaluate your trained policies with the following command:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

Find the `wandb_path` in the WandB run overview.

## Code Structure Overview

The code is structured to promote modularity and ease of use. Key directories and files include:

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**: Contains core functions for the MDP (Markov Decision Process).
    *   `commands.py`: Computes relevant variables from reference motion, current robot state, and error.
    *   `rewards.py`: Implements DeepMimic reward functions and smoothing.
    *   `events.py`: Implements domain randomization.
    *   `observations.py`: Implements observation terms.
    *   `terminations.py`: Implements early terminations and timeouts.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO (Proximal Policy Optimization) hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings.
*   **`scripts`**: Utility scripts for data preprocessing, training, and evaluation.