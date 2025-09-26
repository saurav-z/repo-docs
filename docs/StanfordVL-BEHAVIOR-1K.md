# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark

[![BEHAVIOR-1K Logo](docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K is a comprehensive simulation benchmark enabling researchers to train and evaluate embodied AI agents on 1,000 realistic, everyday household activities.** This repository provides everything you need to get started, including the environment, tasks, and evaluation tools, making it ideal for advancing the field of embodied AI.  Explore the full scope of the project on the [main website](https://behavior.stanford.edu/).

**[View the original repository on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K)**

## Key Features:

*   **1,000 Everyday Activities:** Test your agents on a vast array of tasks, selected from real-world human time-use surveys and preference studies.
*   **Human-Centered Focus:**  Designed for tasks like cleaning, cooking, and organizing, mirroring human daily life.
*   **Comprehensive Simulation:**  Includes a complete environment for training and evaluating agents.
*   **Modular Installation:** Install only the components you need with the flexible setup script.
*   **Realistic Simulations:** Leveraging the OmniGibson physics simulator.

## Installation Guide

Get started quickly with these installation steps:

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quickstart Installation (Recommended)

Install the latest stable release (v3.7.1) with all components.

**Linux:**

```bash
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

**Windows:**

```powershell
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

**Note for Windows:** Run PowerShell as Administrator and set the execution policy if necessary: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation using the following components and options:

#### Available Components

| Component      | Flag           | Description                                          |
| -------------- | -------------- | ---------------------------------------------------- |
| OmniGibson     | `--omnigibson` | Core physics simulator and robotics environment    |
| BDDL           | `--bddl`       | Behavior Domain Definition Language for task specification |
| JoyLo          | `--joylo`      | JoyLo interface for robot teleoperation               |

#### Additional Options

| Option              | Flag                      | Description                                                                     |
| ------------------- | ------------------------- | ------------------------------------------------------------------------------- |
| New Environment     | `--new-env`               | Create a new conda environment named `behavior` (requires conda)                  |
| Datasets            | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                           |
| Primitives          | `--primitives`            | Install OmniGibson with action primitives support                                 |
| Evaluation          | `--eval`                  | Install evaluation support for OmniGibson                                      |
| Development         | `--dev`                   | Install development dependencies                                                |
| CUDA Version        | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                           |
| No Conda Confirmation | `--confirm-no-conda`      | Skip confirmation prompt when not in a conda environment                       |
| Conda TOS           | `--accept-conda-tos`      | Automatically accept Anaconda Terms of Service (for automated installations)     |
| NVIDIA EULA         | `--accept-nvidia-eula`    | Automatically accept NVIDIA Isaac Sim End User License Agreement (for automated installations) |
| Dataset License     | `--accept-dataset-tos`    | Automatically accept BEHAVIOR Data Bundle License Agreement (for automated installations) |

### Installation Without Conda

If you prefer to use an existing Python environment (venv, etc.), omit the `--new-env` flag:

**Linux:**

```bash
./setup.sh --omnigibson --bddl --joylo --dataset
```

**Windows:**

```powershell
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

### Terms of Service & License Acceptance

For automated or CI/CD environments, use the following flags to accept licenses:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

View all available options:
```bash
./setup.sh --help
```

## Citation

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```