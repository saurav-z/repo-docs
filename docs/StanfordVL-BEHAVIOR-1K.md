# BEHAVIOR-1K: Your Gateway to Embodied AI for Everyday Tasks

![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)

**BEHAVIOR-1K is a comprehensive benchmark and simulator, enabling researchers to train and evaluate embodied AI agents on 1,000 realistic, human-centered activities.** This repository provides everything you need to get started with tasks like cleaning, cooking, and organizing, drawn directly from real-world human behavior and preferences.  Check out the [main website](https://behavior.stanford.edu/) for more details!  For the original source code, please visit the [GitHub repository](https://github.com/StanfordVL/BEHAVIOR-1K).

## Key Features

*   **1,000 Everyday Activities:** Test your AI agents on a vast range of tasks inspired by real-world human behavior.
*   **Realistic Simulation:** Built on the robust OmniGibson physics simulator, providing a high-fidelity environment for training and evaluation.
*   **Modular Installation:**  Install only the components you need for a streamlined setup.
*   **Human-Centered Focus:** Designed around activities derived from human time-use surveys and preference studies.
*   **Comprehensive Documentation:** Includes detailed installation instructions, options, and citation information.

## Installation Guide

Get started with BEHAVIOR-1K by following these simple installation steps.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start: Recommended Installation (Latest Stable Release)

This is the easiest way to get up and running.

**Linux:**

```bash
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

**Windows (PowerShell as Administrator):**

```powershell
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

> **Development Branch:** For the latest features (potentially less stable), clone the `main` branch:
> ```bash
> git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
> ```

> **Note**:  On Windows, run PowerShell as Administrator and potentially set the execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation with these components and options:

#### Available Components

| Component        | Flag          | Description                                         |
|-----------------|---------------|-----------------------------------------------------|
| OmniGibson      | `--omnigibson` | Core physics simulator and robotics environment      |
| BDDL            | `--bddl`      | Behavior Domain Definition Language for task specification |
| JoyLo           | `--joylo`     | JoyLo interface for robot teleoperation             |

#### Additional Options

| Option                   | Flag                     | Description                                                                           |
|--------------------------|--------------------------|---------------------------------------------------------------------------------------|
| New Environment          | `--new-env`              | Create a new conda environment named `behavior` (requires conda)                      |
| Datasets                 | `--dataset`              | Download BEHAVIOR datasets (requires `--omnigibson`)                                  |
| Primitives               | `--primitives`           | Install OmniGibson with action primitives support                                     |
| Eval                     | `--eval`                 | Install evaluation support for OmniGibson                                            |
| Development              | `--dev`                  | Install development dependencies                                                      |
| CUDA Version             | `--cuda-version X.X`     | Specify CUDA version (default: 12.4)                                                |
| No Conda Confirmation    | `--confirm-no-conda`     | Skip confirmation prompt when not in a conda environment                               |
| Accept Conda TOS         | `--accept-conda-tos`     | Automatically accept Anaconda Terms of Service                                       |
| Accept NVIDIA EULA      | `--accept-nvidia-eula`  | Automatically accept NVIDIA Isaac Sim End User License Agreement                        |
| Accept Dataset License   | `--accept-dataset-tos`  | Automatically accept BEHAVIOR Data Bundle License Agreement                           |

### Installation without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag:

**Linux:**

```bash
./setup.sh --omnigibson --bddl --joylo --dataset
```

**Windows:**

```powershell
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

To skip the confirmation prompt when not in a conda environment (useful for CI/CD):

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
```

### Terms of Service & License Acceptance (Automated Installation)

For automated or CI/CD environments, use these flags to accept required licenses:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To view all installation options:

```bash
./setup.sh --help
```

## Citation

If you use BEHAVIOR-1K in your research, please cite the following paper:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}