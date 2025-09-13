# BEHAVIOR-1K: Embodied AI for Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**Tackle the complexities of human-centered robotics with BEHAVIOR-1K, a comprehensive benchmark simulating 1,000 everyday household activities.** This repository provides everything you need to train and evaluate embodied AI agents on tasks like cleaning, cooking, and organizing, reflecting real-world human behaviors.  Explore the full details on the [main website](https://behavior.stanford.edu/).

## Key Features

*   **1,000 Everyday Activities:**  A vast and diverse set of tasks drawn from real-world human time-use surveys and preference studies.
*   **Realistic Simulation:** Built upon the robust OmniGibson physics simulator, enabling agents to interact with the environment.
*   **Comprehensive Benchmark:**  Provides a complete solution for training, evaluating, and comparing embodied AI agents.
*   **Modular Installation:**  Install only the components you need with the flexible setup script.
*   **Human-Centered Tasks:** Focuses on activities like cleaning, cooking, and organizing, making it highly relevant to practical applications.

## Installation Guide

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

For the easiest setup, install the latest stable release (v3.7.1) with all components:

**Linux:**

```bash
# Clone the latest stable release (recommended)
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

**Windows:**

```powershell
# Clone the latest stable release (recommended)
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

> **Development Branch:** For the latest features, potentially less stable, clone the `main` branch:
> ```bash
> git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
> ```

> **Note:** Run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

The setup script provides several components and options to tailor the installation.

#### Available Components

| Component        | Flag          | Description                                     |
|------------------|---------------|-------------------------------------------------|
| **OmniGibson**   | `--omnigibson` | Core physics simulator and robotics environment |
| **BDDL**         | `--bddl`       | Behavior Domain Definition Language for task specification |
| **JoyLo**        | `--joylo`      | JoyLo interface for robot teleoperation         |

#### Additional Options

| Option                  | Flag                       | Description                                                                            |
|-------------------------|----------------------------|----------------------------------------------------------------------------------------|
| **New Environment**     | `--new-env`                | Create a new conda environment named `behavior` (requires conda)                       |
| **Datasets**            | `--dataset`                | Download BEHAVIOR datasets (requires `--omnigibson`)                                     |
| **Primitives**          | `--primitives`             | Install OmniGibson with action primitives support                                      |
| **Eval**                | `--eval`                   | Install evaluation support for OmniGibson                                               |
| **Development**         | `--dev`                    | Install development dependencies                                                         |
| **CUDA Version**        | `--cuda-version X.X`       | Specify CUDA version (default: 12.4)                                                    |
| **No Conda Confirmation** | `--confirm-no-conda`       | Skip confirmation prompt when not in a conda environment                              |
| **Conda TOS**           | `--accept-conda-tos`       | Automatically accept Anaconda Terms of Service                                        |
| **NVIDIA EULA**         | `--accept-nvidia-eula`     | Automatically accept NVIDIA Isaac Sim End User License Agreement                        |
| **Dataset License**     | `--accept-dataset-tos`     | Automatically accept BEHAVIOR Data Bundle License Agreement                            |

### Installation without Conda

If you have an existing Python environment, omit the `--new-env` flag:

**Linux:**

```bash
./setup.sh --omnigibson --bddl --joylo --dataset
```

**Windows:**

```powershell
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

To skip the confirmation prompt when not using conda, use the `--confirm-no-conda` flag.

### Automated Installation & License Acceptance

For automated or CI/CD environments, include the following flags to accept all necessary terms:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

For a full list of options, run:

```bash
./setup.sh --help
```

## Citation

If you use BEHAVIOR-1K in your research, please cite the following:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```

**[Back to Top](#behavior-1k-embodied-ai-for-everyday-activities)**