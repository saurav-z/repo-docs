<h1 align="center">BEHAVIOR-1K: Embodied AI Benchmark for Everyday Activities</h1>

![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)

**Tackle real-world challenges in embodied AI with BEHAVIOR-1K, a comprehensive benchmark featuring 1,000 household activities for training and evaluating intelligent agents.**  This repository provides everything you need to get started.

➡️ **[Explore the BEHAVIOR-1K website](https://behavior.stanford.edu/) for more details and resources.**

## Key Features of BEHAVIOR-1K:

*   **Comprehensive Activity Set:**  Tests agents on 1,000 diverse everyday household activities.
*   **Human-Centered Design:** Activities are selected from real-world time-use surveys and preference studies.
*   **Monolithic Repository:** Includes all necessary tools for training and evaluation.
*   **Modular Installation:**  Install only the components you need for a streamlined setup.
*   **Realistic Simulation:** Powered by the OmniGibson physics simulator.

## 🛠️ Installation Guide

Get up and running with BEHAVIOR-1K using our easy-to-use installation script.

### System Requirements

*   **Operating System:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ (Recommended)
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start Installation (Recommended)

Install the latest stable release (v3.7.1) with all components for the best experience.

**Linux:**

```bash
# Clone the repository
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

**Windows (PowerShell):**

```powershell
# Clone the repository
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

**Note:** If you prefer the latest development features (potentially less stable), clone the `main` branch instead: `git clone https://github.com/StanfordVL/BEHAVIOR-1K.git`

> **Important Windows Note:**  Run PowerShell as Administrator and configure the execution policy if necessary: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation with these component flags:

**Available Components:**

| Component       | Flag            | Description                                        |
| --------------- | --------------- | -------------------------------------------------- |
| OmniGibson      | `--omnigibson`  | Core physics simulator and robotics environment    |
| BDDL            | `--bddl`        | Behavior Domain Definition Language for task spec. |
| JoyLo           | `--joylo`       | Interface for robot teleoperation                  |

**Additional Options:**

| Option                      | Flag                       | Description                                                                |
| --------------------------- | -------------------------- | -------------------------------------------------------------------------- |
| New Environment             | `--new-env`                | Creates a new Conda environment named `behavior` (requires Conda)          |
| Datasets                    | `--dataset`                | Downloads BEHAVIOR datasets (requires `--omnigibson`)                      |
| Action Primitives           | `--primitives`             | Installs OmniGibson with action primitives support                          |
| Evaluation Support          | `--eval`                   | Installs evaluation support for OmniGibson                                  |
| Development Dependencies    | `--dev`                    | Installs development dependencies                                           |
| CUDA Version                | `--cuda-version X.X`       | Specifies the CUDA version (default: 12.4)                                  |
| Skip Conda Confirmation   | `--confirm-no-conda`     | Skips the confirmation prompt when not in a Conda environment             |
| Accept Conda TOS          | `--accept-conda-tos`     | Automatically accepts Anaconda Terms of Service                                  |
| Accept NVIDIA EULA        | `--accept-nvidia-eula`     | Automatically accepts NVIDIA Isaac Sim End User License Agreement                                  |
| Accept Dataset License   | `--accept-dataset-tos`     | Automatically accepts BEHAVIOR Data Bundle License Agreement                                  |

### Installation Without Conda

If you prefer to use your existing Python environment (system Python, venv, etc.), omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --joylo --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

To bypass confirmation prompts, especially useful for CI/CD:

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

For a full list of installation options, run:
```bash
./setup.sh --help
```

## 📖 Citation

If you use BEHAVIOR-1K, please cite the following paper:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```

➡️ **[View the original repository on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K)**