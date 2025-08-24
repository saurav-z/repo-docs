# BEHAVIOR-1K: A Benchmark for Embodied AI in Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K empowers researchers to develop and evaluate embodied AI agents by providing a comprehensive simulation environment for 1,000 realistic household tasks.**  This repository contains everything needed to train and assess agents on a diverse range of human-centered activities, including cleaning, cooking, and organization, derived from real-world human activity data.

[Explore the full details on our main website!](https://behavior.stanford.edu/)

## Key Features

*   **Extensive Task Coverage:** Simulate and evaluate agents on 1,000 diverse everyday activities.
*   **Realistic Simulation:** Powered by OmniGibson, a robust physics simulator and robotics environment.
*   **Human-Centered Focus:** Tasks are based on human time-use surveys and preference studies, reflecting real-world scenarios.
*   **Modular Installation:**  Install only the components you need to get started quickly.
*   **Teleoperation Support:** Includes JoyLo interface for robot teleoperation.

## Installation Guide

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

Follow these steps for a complete installation (recommended):

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

> **Important:**  On Windows, run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

BEHAVIOR-1K offers flexible installation options.

#### Available Components

| Component       | Flag            | Description                                             |
|-----------------|-----------------|---------------------------------------------------------|
| OmniGibson      | `--omnigibson`  | Core physics simulator and robotics environment        |
| BDDL            | `--bddl`        | Behavior Domain Definition Language for task specification |
| Teleoperation   | `--teleop`      | JoyLo interface for robot teleoperation                 |

#### Additional Installation Options

| Option                     | Flag                         | Description                                                                                                     |
|----------------------------|------------------------------|-----------------------------------------------------------------------------------------------------------------|
| New Environment            | `--new-env`                  | Create a new conda environment named `behavior` (requires conda)                                                |
| Datasets                   | `--dataset`                  | Download BEHAVIOR datasets (requires `--omnigibson`)                                                              |
| Primitives                 | `--primitives`               | Install OmniGibson with action primitives support                                                               |
| Development                | `--dev`                      | Install development dependencies                                                                                |
| CUDA Version               | `--cuda-version X.X`         | Specify CUDA version (default: 12.4)                                                                             |
| No Conda Confirmation      | `--confirm-no-conda`         | Skip confirmation prompt when not in a conda environment                                                        |
| Accept Conda TOS           | `--accept-conda-tos`         | Automatically accept Anaconda Terms of Service                                                                  |
| Accept NVIDIA EULA         | `--accept-nvidia-eula`       | Automatically accept NVIDIA Isaac Sim End User License Agreement                                               |
| Accept Dataset License     | `--accept-dataset-tos`       | Automatically accept BEHAVIOR Data Bundle License Agreement                                                     |

### Installation without Conda

If you prefer to use an existing Python environment, exclude `--new-env`:

```bash
# Linux
./setup.sh --omnigibson --bddl --teleop --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

### Automating Installation (e.g., CI/CD)

To automate the installation and bypass all prompts, use:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

### Help

To view all available installation options:
```bash
./setup.sh --help
```

## Citation

If you use BEHAVIOR-1K, please cite our work:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```