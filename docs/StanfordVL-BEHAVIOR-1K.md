# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark for Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K** is a groundbreaking simulation benchmark designed to challenge and advance embodied AI agents by testing their abilities to perform 1,000 realistic household tasks.  Explore the full details and resources on the [original GitHub repository](https://github.com/StanfordVL/BEHAVIOR-1K).

***Visit our [main website](https://behavior.stanford.edu/) for more information!***

## Key Features of BEHAVIOR-1K:

*   **Comprehensive Benchmark:** Evaluates agents on a diverse set of 1,000 everyday activities, mirroring real-world human behaviors.
*   **Human-Centered Tasks:** Focuses on tasks derived from human time-use surveys and preference studies, ensuring relevance and practicality.
*   **Modular Installation:** Offers a flexible installation process, allowing users to install only the necessary components.
*   **Realistic Simulation:** Utilizes the OmniGibson physics simulator to create a realistic and interactive environment.
*   **Teleoperation Interface:** Includes a JoyLo interface for robot teleoperation.
*   **BDDL Integration:** Leverages the Behavior Domain Definition Language for task specification.

## Installation Guide

This section provides detailed instructions for setting up the BEHAVIOR-1K environment.

### System Requirements

*   **OS**: Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM**: 32GB+ recommended
*   **VRAM**: 8GB+
*   **GPU**: NVIDIA RTX 2080+

### Quick Start Installation (Recommended)

These commands will install all the necessary components for most users.  Choose either the Linux or Windows command based on your operating system.

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

> **Note:**  On Windows, run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation using the following flags:

#### Available Components

| Component       | Flag            | Description                                         |
|-----------------|-----------------|-----------------------------------------------------|
| **OmniGibson**  | `--omnigibson`  | Core physics simulator and robotics environment      |
| **BDDL**        | `--bddl`        | Behavior Domain Definition Language for task specification |
| **Teleoperation** | `--teleop`      | JoyLo interface for robot teleoperation          |

#### Additional Options

| Option                 | Flag                     | Description                                                                   |
|------------------------|--------------------------|-------------------------------------------------------------------------------|
| **New Environment**    | `--new-env`              | Create a new conda environment named `behavior` (requires conda)                |
| **Datasets**           | `--dataset`              | Download BEHAVIOR datasets (requires `--omnigibson`)                           |
| **Primitives**         | `--primitives`           | Install OmniGibson with action primitives support                              |
| **Development**        | `--dev`                  | Install development dependencies                                               |
| **CUDA Version**       | `--cuda-version X.X`     | Specify CUDA version (default: 12.4)                                         |
| **No Conda Confirmation** | `--confirm-no-conda`    | Skip confirmation prompt when not in a conda environment                      |
| **Conda TOS**          | `--accept-conda-tos`     | Automatically accept Anaconda Terms of Service                               |
| **NVIDIA EULA**        | `--accept-nvidia-eula`   | Automatically accept NVIDIA Isaac Sim End User License Agreement               |
| **Dataset License**    | `--accept-dataset-tos`   | Automatically accept BEHAVIOR Data Bundle License Agreement                   |

### Installation without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --teleop --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

Skip the confirmation prompt (useful for CI/CD):

```bash
./setup.sh --omnigibson --bddl --teleop --dataset --confirm-no-conda
```

### Terms of Service & License Acceptance

Accepting necessary terms for automated installation in CI/CD environments:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To view all available options:

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