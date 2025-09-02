# BEHAVIOR-1K: Unleash Embodied AI in the Real World

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K** is a comprehensive benchmark pushing the boundaries of embodied AI by challenging agents to complete 1,000 realistic household tasks.

Explore the full potential of embodied AI and dive into the world of cleaning, cooking, and organizing with this powerful simulation. For more in-depth information, please visit our [main website](https://behavior.stanford.edu/).

## Key Features

*   **1,000 Everyday Activities:**  Simulate a wide range of human-centric tasks inspired by real-world time-use surveys and preference studies.
*   **Realistic Simulation:** Train and evaluate agents in a high-fidelity simulation environment.
*   **Modular Installation:** Easily install only the components you need for a streamlined setup.
*   **Comprehensive Dataset:** Access a rich dataset to facilitate training and evaluation.
*   **Teleoperation Support:** Experiment with teleoperation interfaces for hands-on control.
*   **Open Source:** Fully accessible on [GitHub](https://github.com/StanfordVL/BEHAVIOR-1K)

## Installation Guide

This section provides instructions for setting up BEHAVIOR-1K.  A modular installation is supported, which allows you to install only the components you require.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

The full installation is recommended for most users. Follow the instructions below for your operating system, creating a new conda environment (or using your existing Python environment).

#### Linux

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --teleop --dataset
```

#### Windows

```powershell
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dataset
```

> **Note**: Run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

#### Available Components

| Component       | Flag          | Description                                         |
| --------------- | ------------- | --------------------------------------------------- |
| **OmniGibson**  | `--omnigibson` | Core physics simulator and robotics environment     |
| **BDDL**        | `--bddl`       | Behavior Domain Definition Language for task specification |
| **Teleoperation** | `--teleop`     | JoyLo interface for robot teleoperation           |

#### Additional Options

| Option                | Flag                      | Description                                                                |
| --------------------- | ------------------------- | -------------------------------------------------------------------------- |
| **New Environment**   | `--new-env`               | Create a new conda environment named `behavior` (requires conda)              |
| **Datasets**          | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                      |
| **Primitives**        | `--primitives`            | Install OmniGibson with action primitives support                          |
| **Development**       | `--dev`                   | Install development dependencies                                           |
| **CUDA Version**      | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                       |
| **No Conda Confirmation** | `--confirm-no-conda` | Skip confirmation prompt when not in a conda environment                      |
| **Conda TOS**         | `--accept-conda-tos`       | Automatically accept Anaconda Terms of Service                            |
| **NVIDIA EULA**       | `--accept-nvidia-eula`    | Automatically accept NVIDIA Isaac Sim End User License Agreement          |
| **Dataset License**   | `--accept-dataset-tos`    | Automatically accept BEHAVIOR Data Bundle License Agreement                 |

### Installation without Conda

If you prefer to use your existing Python environment (system Python, venv, etc.) instead of conda, simply omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --teleop --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

If you're not in a conda environment, the script will prompt for confirmation. To skip this prompt (useful for CI/CD):

```bash
./setup.sh --omnigibson --bddl --teleop --dataset --confirm-no-conda
```

### Terms of Service & License Acceptance

BEHAVIOR-1K installation may require acceptance of various terms of service and license agreements. For interactive installation, you'll be prompted to accept these terms. For non-interactive/automated installation, use these flags:

For automated/CI environments, you can bypass all prompts:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To see all available options:
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
```