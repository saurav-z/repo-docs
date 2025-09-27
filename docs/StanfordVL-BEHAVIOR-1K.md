# BEHAVIOR-1K: Embodied AI for Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**Tackle the complexities of everyday life with BEHAVIOR-1K, a comprehensive benchmark for embodied AI agents, simulating 1,000 real-world household tasks.**

[Explore the main website for more details](https://behavior.stanford.edu/)

## Key Features

*   **1,000 Everyday Activities:** Test AI agents on a vast range of human-centered tasks, including cleaning, cooking, and organizing.
*   **Realistic Simulation:** Built on the OmniGibson physics simulator for a high-fidelity environment.
*   **Human-Centered Tasks:** Activities selected from real-world time-use surveys and preference studies.
*   **Modular Installation:** Easily install only the components you need with a flexible setup script.

## Installation Guide

This section details the setup process for BEHAVIOR-1K.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start Installation (Recommended)

The following steps outline the recommended installation procedure for the latest stable release.

**Linux:**

```bash
# Clone the latest stable release (v3.7.1)
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

**Windows:**

```powershell
# Clone the latest stable release (v3.7.1)
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

> **Note:** If you're interested in the latest, potentially less stable, features, clone the `main` branch instead.

> **Important for Windows:** Run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

BEHAVIOR-1K provides flexibility in selecting components and configuring the installation process.

#### Available Components

| Component      | Flag           | Description                                                |
|----------------|----------------|------------------------------------------------------------|
| OmniGibson     | `--omnigibson` | Core physics simulator and robotics environment            |
| BDDL           | `--bddl`       | Behavior Domain Definition Language for task specification |
| JoyLo          | `--joylo`      | JoyLo interface for robot teleoperation                    |

#### Additional Installation Options

| Option                  | Flag                         | Description                                                                           |
|-------------------------|------------------------------|---------------------------------------------------------------------------------------|
| New Environment         | `--new-env`                  | Create a new conda environment named `behavior` (requires conda)                       |
| Datasets                | `--dataset`                  | Download BEHAVIOR datasets (requires `--omnigibson`)                                 |
| Primitives              | `--primitives`               | Install OmniGibson with action primitives support                                   |
| Evaluation Support      | `--eval`                     | Install evaluation support for OmniGibson                                           |
| Development Dependencies | `--dev`                      | Install development dependencies                                                      |
| CUDA Version            | `--cuda-version X.X`         | Specify CUDA version (default: 12.4)                                                 |
| No Conda Confirmation  | `--confirm-no-conda`         | Skip confirmation prompt when not in a conda environment                             |
| Accept Conda TOS        | `--accept-conda-tos`         | Automatically accept Anaconda Terms of Service                                       |
| Accept NVIDIA EULA      | `--accept-nvidia-eula`       | Automatically accept NVIDIA Isaac Sim End User License Agreement                       |
| Accept Dataset License  | `--accept-dataset-tos`       | Automatically accept BEHAVIOR Data Bundle License Agreement                            |

#### Installation Without Conda

If you prefer to use your existing Python environment (e.g., system Python, venv), exclude the `--new-env` flag.

```bash
# Linux
./setup.sh --omnigibson --bddl --joylo --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

For automated installations without prompts, use the acceptance flags:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

Get more information on all available installation options:

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