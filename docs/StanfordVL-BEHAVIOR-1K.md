# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark for Everyday Tasks

[![BEHAVIOR-1K](docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K** offers a comprehensive simulated environment for training and evaluating embodied AI agents on 1,000 realistic household activities, mimicking real-world human behavior. ([Original Repository](https://github.com/StanfordVL/BEHAVIOR-1K))

**Key Features:**

*   **1,000 Everyday Activities:** Focuses on human-centered tasks like cooking, cleaning, and organizing, mirroring real-world time-use data.
*   **Realistic Simulation:** Leverages OmniGibson for high-fidelity physics simulation and robotics environment.
*   **Modular Installation:** Install only the components you need with a flexible setup script.
*   **Comprehensive Dataset:** Provides datasets for training and evaluation.
*   **Easy to Use:** Includes a streamlined installation process for both Linux and Windows.

## Installation Guide

This section provides detailed instructions for installing BEHAVIOR-1K and its dependencies.

### System Requirements

*   **Operating System:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start Installation (Recommended)

This method installs the latest stable release with all core components.

**Linux:**

```bash
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

**Windows:**

```powershell
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

**Development Branch:** For the latest features (potentially less stable), clone the `main` branch:

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

**Important Notes for Windows:**

*   Run PowerShell as Administrator.
*   Set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Advanced Installation Options

### Available Components

The installation script supports various components to customize your setup:

| Component       | Flag            | Description                                         |
|-----------------|-----------------|-----------------------------------------------------|
| OmniGibson      | `--omnigibson`  | Core physics simulator and robotics environment     |
| BDDL            | `--bddl`        | Behavior Domain Definition Language for task spec  |
| JoyLo           | `--joylo`       | JoyLo interface for robot teleoperation            |

### Additional Installation Options

Customize your installation with these options:

| Option                     | Flag                       | Description                                                                |
|----------------------------|----------------------------|----------------------------------------------------------------------------|
| New Environment            | `--new-env`                | Create a new conda environment named `behavior` (requires conda)           |
| Datasets                   | `--dataset`                | Download BEHAVIOR datasets (requires `--omnigibson`)                        |
| Action Primitives          | `--primitives`             | Install OmniGibson with action primitives support                         |
| Evaluation Support         | `--eval`                   | Install evaluation support for OmniGibson                                  |
| Development Dependencies   | `--dev`                    | Install development dependencies                                             |
| CUDA Version               | `--cuda-version X.X`       | Specify CUDA version (default: 12.4)                                       |
| Skip Conda Confirmation  | `--confirm-no-conda`       | Skip confirmation prompt when not in a conda environment                     |
| Accept Conda TOS            | `--accept-conda-tos`       | Automatically accept Anaconda Terms of Service                                 |
| Accept NVIDIA EULA         | `--accept-nvidia-eula`     | Automatically accept NVIDIA Isaac Sim End User License Agreement              |
| Accept Dataset License     | `--accept-dataset-tos`     | Automatically accept BEHAVIOR Data Bundle License Agreement                  |

### Installation Without Conda

If you prefer to use an existing Python environment, omit the `--new-env` flag:

**Linux:**

```bash
./setup.sh --omnigibson --bddl --joylo --dataset
```

**Windows:**

```powershell
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

**Skipping Confirmation:** To skip the prompt when not in a conda environment:

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
```

### Automated/CI Installation

For automated installations, accept all required licenses:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

### Get Help

To view all available options:

```bash
./setup.sh --help
```

## Citation

If you use BEHAVIOR-1K, please cite the following paper:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```