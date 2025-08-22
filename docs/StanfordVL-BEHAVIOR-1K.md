# BEHAVIOR-1K: The Ultimate Benchmark for Embodied AI in Everyday Life

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K** is a comprehensive simulation benchmark designed to evaluate and advance embodied AI agents on a wide range of human-centered activities, providing a realistic and challenging environment for training and testing.  Check out the [original repository on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K) for more details and the latest updates.

**Key Features:**

*   **1,000 Everyday Activities:** Tests agents on tasks inspired by real-world human time-use surveys, including cleaning, cooking, and organization.
*   **Realistic Simulation:** Leverages advanced simulation capabilities to provide a challenging and immersive environment.
*   **Modular Installation:**  Install only the components you need for maximum flexibility.
*   **Human-Centered Focus:**  Addresses the need for AI agents capable of assisting with everyday tasks in a human-centric way.
*   **Extensive Datasets:** Includes datasets to support training and evaluation of your AI agents.
*   **Teleoperation Support:**  Allows for human teleoperation to interact with the simulated environment.

## Installation

The BEHAVIOR-1K project offers a convenient installation script to handle all dependencies.  This script is designed for ease of use and allows for modular installation of specific components.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

For most users, a full installation is recommended. Choose to create a new conda environment or use your existing Python environment.

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

> **Note:** Run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

#### Available Components

| Component        | Flag           | Description                                                    |
| ---------------- | -------------- | -------------------------------------------------------------- |
| OmniGibson       | `--omnigibson` | Core physics simulator and robotics environment                  |
| BDDL             | `--bddl`       | Behavior Domain Definition Language for task specification     |
| Teleoperation    | `--teleop`     | JoyLo interface for robot teleoperation                        |

#### Additional Options

| Option                     | Flag                    | Description                                                                |
| -------------------------- | ----------------------- | -------------------------------------------------------------------------- |
| New Environment            | `--new-env`             | Create a new conda environment named `behavior` (requires conda)          |
| Datasets                   | `--dataset`             | Download BEHAVIOR datasets (requires `--omnigibson`)                        |
| Primitives                 | `--primitives`          | Install OmniGibson with action primitives support                             |
| Development                | `--dev`                 | Install development dependencies                                          |
| CUDA Version               | `--cuda-version X.X`    | Specify CUDA version (default: 12.4)                                       |
| No Conda Confirmation      | `--confirm-no-conda`    | Skip confirmation prompt when not in a conda environment                   |
| Conda TOS                  | `--accept-conda-tos`    | Automatically accept Anaconda Terms of Service                             |
| NVIDIA EULA                | `--accept-nvidia-eula`  | Automatically accept NVIDIA Isaac Sim End User License Agreement          |
| Dataset License            | `--accept-dataset-tos`  | Automatically accept BEHAVIOR Data Bundle License Agreement                 |

To see all available options:
```bash
./setup.sh --help
```

### Installation without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --teleop --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

To skip the confirmation prompt when not in a conda environment, use the `--confirm-no-conda` flag.

### Terms of Service & License Acceptance

For automated/CI environments, you can bypass all prompts using the following flags:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
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