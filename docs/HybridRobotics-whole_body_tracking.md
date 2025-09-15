# BeyondMimic: High-Fidelity Humanoid Motion Tracking with Guided Diffusion

**BeyondMimic empowers you to achieve state-of-the-art humanoid motion tracking and control, bridging the gap between simulation and real-world deployment.** Explore the full potential of this framework at the original repository: [https://github.com/HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking).

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

*   **Sim-to-Real Ready:** Train motion tracking models ready for real-world deployment.
*   **LAFAN1 Dataset Support:** Train on the LAFAN1 dataset with minimal parameter tuning.
*   **Guided Diffusion-Based Controllers:** Utilize steerable test-time control for advanced motion control.
*   **WandB Integration:** Leverage Weights & Biases for easy motion management and experiment tracking.

## Installation

1.  **Install Isaac Lab:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for Isaac Lab v2.1.0. Conda installation is recommended.
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
4.  **Install the Library:** Using a Python interpreter with Isaac Lab installed:
    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## Motion Tracking

### Motion Preprocessing & Registry Setup

1.  **Gather Motion Datasets:** Acquire reference motion datasets, adhering to their licenses.  Support for LAFAN1 (Hugging Face), Sidekicks (KungfuBot), Christiano Ronaldo (ASAP), and Balance motions (HuB) is included.
2.  **WandB Registry Setup:**
    *   Log in to your WandB account.
    *   Create a new registry collection named "Motions" with artifact type "All Types".
3.  **Convert Motions:** Convert retargeted motions to include maximum coordinate information:
    ```bash
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
    ```
    This uploads the processed file to the WandB registry.
4.  **Test the Registry:** Verify by replaying a motion in Isaac Sim:
    ```bash
    python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
    ```
5.  **Debugging:**
    *   Ensure `WANDB_ENTITY` is set to your organization name.
    *   If the `/tmp` folder isn't accessible, adjust `csv_to_npz.py` to use an alternative temporary folder.

### Policy Training

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```
    Locate the WandB run path in the run overview, typically in the format `{your_organization}/{project_name}/` followed by an 8-character identifier.  Note that `run_name` differs from `run_path`.

## Code Structure Overview

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**:  Core MDP components.

    *   `commands.py`: Command library for error calculations, randomization, and adaptive sampling.
    *   `rewards.py`: DeepMimic reward functions and smoothing terms.
    *   `events.py`: Domain randomization terms.
    *   `observations.py`: Motion tracking and data collection observations.
    *   `terminations.py`: Early termination and timeout conditions.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**: Environment (MDP) configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**: PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`**: Robot-specific settings (armature parameters, joint calculations).
*   **`scripts`**: Utility scripts for data preprocessing, policy training, and evaluation.