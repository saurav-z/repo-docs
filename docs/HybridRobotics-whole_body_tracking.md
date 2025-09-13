# BeyondMimic: Train Realistic Humanoid Motion Tracking in Isaac Sim

**BeyondMimic offers a cutting-edge humanoid control framework for training dynamic motion tracking, enabling sim-to-real readiness and steerable test-time control.** ([Original Repository](https://github.com/HybridRobotics/whole_body_tracking))

## Key Features:

*   **Sim-to-Real Ready:** Train motion tracking models compatible for real-world deployment.
*   **State-of-the-Art Motion Quality:** Achieve highly dynamic and realistic motion tracking.
*   **LAFAN1 Dataset Compatibility:** Train motion on the LAFAN1 dataset with no parameter tuning needed.
*   **WandB Registry Integration:**  Leverages WandB for efficient motion data management and retrieval.
*   **Modular Code Structure:**  Organized codebase for easy modification and expansion.

## Installation

1.  **Install Isaac Lab:** Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for Isaac Lab v2.1.0 (Conda recommended).
2.  **Clone the Repository:**

```bash
git clone https://github.com/HybridRobotics/whole_body_tracking.git
cd whole_body_tracking
```

3.  **Get Robot Description Files:**

```bash
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

4.  **Install the Library:**
    Using your Isaac Lab Python environment:
```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Tracking Workflow

### Motion Preprocessing & Registry Setup

1.  **Gather Datasets:**  Acquire reference motion datasets (LAFAN1, Sidekicks, Christiano Ronaldo celebration, Balance motions).
2.  **WandB Registry Setup:**
    *   Log in to your WandB account.
    *   Create a new registry collection named "Motions" with artifact type "All Types".
3.  **Convert Motions:**

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
```

This generates and uploads processed motion files to your WandB registry.

4.  **Test Registry:**

```bash
python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
```

*   **Debugging:**  Ensure `WANDB_ENTITY` is set to your organization name. If `/tmp` is inaccessible, adjust `csv_to_npz.py` for temporary folder.

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

*   Find the `wandb_path` in your WandB run overview (format: `{your_organization}/{project_name}/{unique_id}`).

## Code Structure

*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`:** Core MDP components (commands, rewards, events, observations, terminations).
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`:** Environment configuration.
*   **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`:** PPO hyperparameters.
*   **`source/whole_body_tracking/whole_body_tracking/robots`:** Robot-specific settings.
*   **`scripts`:** Utility scripts for data preprocessing, training, and evaluation.