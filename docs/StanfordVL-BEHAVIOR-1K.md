# BEHAVIOR-1K: Embodied AI for Everyday Activities

[![BEHAVIOR-1K](docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K is a groundbreaking benchmark for embodied AI, challenging agents to master 1,000 real-world household tasks like cooking and cleaning.** This comprehensive simulation environment provides everything needed to train and evaluate AI agents on complex, human-centric activities.  Explore the official website for more details: [https://behavior.stanford.edu/](https://behavior.stanford.edu/)

**[View the original repository on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K)**

## Key Features

*   **1,000 Everyday Activities:** Train and evaluate agents on a vast range of household tasks.
*   **Human-Centered Design:** Activities are based on real-world time-use surveys and preference studies.
*   **Realistic Simulation:** Leverage the OmniGibson physics simulator for accurate and engaging environments.
*   **Modular Installation:** Customize your installation with only the necessary components.
*   **Comprehensive Support:** Includes BDDL for task specification and JoyLo for teleoperation.

## Installation

Get started with BEHAVIOR-1K by following the installation instructions below. The provided scripts streamline the setup process, handling dependencies and component installation.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

For most users, the latest stable release is recommended.

**Linux:**

```bash
# Clone the latest stable release (recommended)
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

**Windows:**

```powershell
# Clone the latest stable release (recommended)
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

> **Development Branch**:  For the latest features (potentially unstable), clone the `main` branch instead.

> **Note**: Windows users should run PowerShell as Administrator and may need to set the execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation using the following components and options:

#### Available Components

| Component        | Flag           | Description                                        |
| ---------------- | -------------- | -------------------------------------------------- |
| OmniGibson       | `--omnigibson` | Core physics simulator and robotics environment |
| BDDL             | `--bddl`       | Behavior Domain Definition Language for task specification |
| JoyLo            | `--joylo`      | JoyLo interface for robot teleoperation            |

#### Additional Options

| Option                    | Flag                    | Description                                                                      |
| ------------------------- | ----------------------- | -------------------------------------------------------------------------------- |
| New Environment           | `--new-env`             | Create a new conda environment named `behavior` (requires conda)                 |
| Datasets                  | `--dataset`             | Download BEHAVIOR datasets (requires `--omnigibson`)                            |
| Primitives                | `--primitives`          | Install OmniGibson with action primitives support                                 |
| Evaluation                | `--eval`                | Install evaluation support for OmniGibson                                         |
| Development Dependencies  | `--dev`                 | Install development dependencies                                                 |
| CUDA Version              | `--cuda-version X.X`    | Specify CUDA version (default: 12.4)                                            |
| No Conda Confirmation     | `--confirm-no-conda`    | Skip confirmation prompt when not in a conda environment                          |
| Accept Conda TOS          | `--accept-conda-tos`    | Automatically accept Anaconda Terms of Service                                    |
| Accept NVIDIA EULA        | `--accept-nvidia-eula`  | Automatically accept NVIDIA Isaac Sim End User License Agreement                  |
| Accept Dataset License    | `--accept-dataset-tos`  | Automatically accept BEHAVIOR Data Bundle License Agreement                       |

### Installation without Conda

To use your existing Python environment (system Python, venv, etc.), exclude the `--new-env` flag:

**Linux:**

```bash
./setup.sh --omnigibson --bddl --joylo --dataset
```

**Windows:**

```powershell
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

If not in a conda environment, the script will prompt for confirmation.  To skip the prompt (useful for CI/CD):

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
```

### Automated Installation and License Acceptance

For automated installations (e.g., CI/CD environments), accept all necessary terms and conditions:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

###  Help and Options

View all available installation options:

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