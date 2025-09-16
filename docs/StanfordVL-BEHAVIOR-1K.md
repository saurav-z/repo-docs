# BEHAVIOR-1K: Embodied AI for Everyday Tasks

[![BEHAVIOR-1K](docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**Tackle the complexities of real-world tasks: BEHAVIOR-1K provides a comprehensive benchmark for training and evaluating embodied AI agents on 1,000 diverse household activities.** This cutting-edge simulation environment empowers researchers to develop and test AI agents in realistic, human-centered scenarios.

**Explore the official repository on GitHub: [https://github.com/StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)**

## Key Features

*   **1,000 Everyday Activities:** Train agents on a vast range of tasks like cooking, cleaning, and organizing, mirroring real-world human behavior.
*   **Realistic Simulation:** Built upon robust physics simulation, providing a true-to-life environment for agent training and evaluation.
*   **Human-Centered Design:** Activities are selected from real-world human time-use surveys and preference studies, ensuring relevance and practicality.
*   **Modular Installation:** Easily install only the components you need with a flexible setup script.

## Installation

Get started with BEHAVIOR-1K by following the simple installation instructions below.  The provided setup script handles all dependencies.

### System Requirements

*   **Operating System:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

For most users, we recommend installing the latest stable release (v3.7.1) with all components:

#### Linux

```bash
# Clone the latest stable release (recommended)
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

#### Windows

```powershell
# Clone the latest stable release (recommended)
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

> **Development Branch:** For the latest features (potentially less stable), clone the main branch instead:
> ```bash
> git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
> ```

> **Note:** Run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation with the following options:

#### Available Components

| Component      | Flag           | Description                                           |
|----------------|----------------|-------------------------------------------------------|
| OmniGibson     | `--omnigibson` | Core physics simulator and robotics environment       |
| BDDL           | `--bddl`       | Behavior Domain Definition Language for task specification |
| JoyLo          | `--joylo`      | JoyLo interface for robot teleoperation               |

#### Additional Options

| Option                    | Flag                       | Description                                                             |
|---------------------------|----------------------------|-------------------------------------------------------------------------|
| New Environment           | `--new-env`                | Create a new conda environment named `behavior` (requires conda)          |
| Datasets                  | `--dataset`                | Download BEHAVIOR datasets (requires `--omnigibson`)                     |
| Primitives                | `--primitives`             | Install OmniGibson with action primitives support                        |
| Evaluation                | `--eval`                   | Install evaluation support for OmniGibson                                |
| Development Dependencies | `--dev`                    | Install development dependencies                                        |
| CUDA Version              | `--cuda-version X.X`       | Specify CUDA version (default: 12.4)                                      |
| No Conda Confirmation     | `--confirm-no-conda`       | Skip confirmation prompt when not in a conda environment                  |
| Accept Conda TOS          | `--accept-conda-tos`       | Automatically accept Anaconda Terms of Service                           |
| Accept NVIDIA EULA        | `--accept-nvidia-eula`     | Automatically accept NVIDIA Isaac Sim End User License Agreement          |
| Accept Dataset License    | `--accept-dataset-tos`     | Automatically accept BEHAVIOR Data Bundle License Agreement             |

### Installation Without Conda

If you prefer to use your existing Python environment (system Python, venv, etc.) instead of conda, omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --joylo --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

To skip confirmation when not in a conda environment:

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
```

## Terms of Service & License Acceptance

Automate your installation with these flags:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To view all available options:
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