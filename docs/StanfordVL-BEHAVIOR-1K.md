# BEHAVIOR-1K: Embodied AI Benchmark for Everyday Household Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**Tackle the complexities of human-centered AI with BEHAVIOR-1K, a comprehensive benchmark simulating 1,000 everyday household activities.** This repository provides everything you need to train and evaluate embodied AI agents on realistic tasks like cleaning, cooking, and organizing, mirroring real-world human behaviors.

**[Explore the official BEHAVIOR-1K website for more details!](https://behavior.stanford.edu/)**

## Key Features

*   **Large-Scale Benchmark:** Test your AI agents on a diverse set of 1,000 activities drawn from real-world human time-use surveys.
*   **Human-Centered Tasks:** Focus on activities like cleaning, cooking, and organizing, providing a realistic and relevant testing ground.
*   **Comprehensive Simulation:** Leverage a fully-featured simulation environment for training and evaluation.
*   **Modular Installation:** Easily install only the components you need with the flexible setup script.

## Installation Guide

Get started with BEHAVIOR-1K by following these simple steps.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ (recommended)
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start (Recommended for Most Users)

Install BEHAVIOR-1K with all core components using a new Conda environment.

**Linux:**

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --teleop --dataset
```

**Windows:**

```powershell
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dataset
```

> **Note:** Run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation with these components and options:

#### Available Components

| Component       | Flag           | Description                                         |
|-----------------|----------------|-----------------------------------------------------|
| OmniGibson      | `--omnigibson` | Core physics simulator and robotics environment     |
| BDDL            | `--bddl`       | Behavior Domain Definition Language for task specification |
| Teleoperation   | `--teleop`     | JoyLo interface for robot teleoperation             |

#### Additional Options

| Option                | Flag                      | Description                                                                       |
|-----------------------|---------------------------|-----------------------------------------------------------------------------------|
| New Environment       | `--new-env`               | Create a new conda environment named `behavior`                                   |
| Datasets              | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                              |
| Primitives            | `--primitives`            | Install OmniGibson with action primitives support                                  |
| Development           | `--dev`                   | Install development dependencies                                                  |
| CUDA Version          | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                              |
| Accept Conda TOS       | `--accept-conda-tos`       | Automatically accept Anaconda Terms of Service                                  |
| Accept NVIDIA EULA     | `--accept-nvidia-eula`     | Automatically accept NVIDIA Isaac Sim End User License Agreement                   |
| Accept Dataset License | `--accept-dataset-tos`     | Automatically accept BEHAVIOR Data Bundle License Agreement                         |

For automated/CI environments, bypass prompts by including acceptance flags for all necessary licenses:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
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

**[Back to the original repository](https://github.com/StanfordVL/BEHAVIOR-1K)**