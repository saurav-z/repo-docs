# BEHAVIOR-1K: Embodied AI for Everyday Activities

**Tackle real-world challenges with BEHAVIOR-1K, a comprehensive benchmark simulating 1,000 household tasks to advance embodied AI.**  [Visit the original repository on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K)

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

BEHAVIOR-1K is a groundbreaking simulation benchmark designed to evaluate and train embodied AI agents on a vast range of everyday household activities. This repository provides everything needed to get started, including tools and datasets focused on tasks such as cleaning, cooking, and organizing, all based on real-world human activity data.

***For more details, explore our [main website](https://behavior.stanford.edu/)!***

## Key Features

*   **Extensive Task Coverage:**  Simulates 1,000 diverse household activities.
*   **Human-Centered Design:** Tasks are selected from real-world time-use surveys and preference studies, making the benchmark highly relevant.
*   **Comprehensive Resources:** Provides all necessary components for training and evaluating embodied AI agents.
*   **Modular Installation:**  Offers flexible installation options to customize your setup.

## Installation

The BEHAVIOR-1K project simplifies setup with a comprehensive installation script. You can choose between a full installation or a modular one based on your specific needs.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

For most users, the full installation is recommended:

**Linux:**

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --teleop --dataset
```

**Windows (PowerShell as Administrator):**

```powershell
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dataset
```

> **Important:** Run PowerShell as Administrator and set the execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

The installation script supports several components and options:

#### Available Components

| Component        | Flag           | Description                                         |
|-----------------|----------------|-----------------------------------------------------|
| OmniGibson      | `--omnigibson` | Core physics simulator and robotics environment      |
| BDDL            | `--bddl`       | Behavior Domain Definition Language for task spec. |
| Teleoperation   | `--teleop`     | JoyLo interface for robot teleoperation             |

#### Additional Options

| Option                  | Flag                         | Description                                                           |
|-------------------------|------------------------------|-----------------------------------------------------------------------|
| New Environment         | `--new-env`                  | Creates a new Conda environment named `behavior` (requires conda)      |
| Datasets                | `--dataset`                  | Downloads BEHAVIOR datasets (requires `--omnigibson`)                  |
| Primitives              | `--primitives`               | Installs OmniGibson with action primitives support                     |
| Development             | `--dev`                      | Installs development dependencies                                       |
| CUDA Version            | `--cuda-version X.X`         | Specifies CUDA version (default: 12.4)                                  |
| No Conda Confirmation   | `--confirm-no-conda`         | Skips confirmation prompt when not in a conda environment               |
| Accept Conda TOS        | `--accept-conda-tos`         | Automatically accepts Anaconda Terms of Service                         |
| Accept NVIDIA EULA      | `--accept-nvidia-eula`       | Automatically accepts NVIDIA Isaac Sim End User License Agreement      |
| Accept Dataset License  | `--accept-dataset-tos`       | Automatically accepts BEHAVIOR Data Bundle License Agreement          |

### Installation without Conda

If you prefer using your existing Python environment, omit the `--new-env` flag:

**Linux:**

```bash
./setup.sh --omnigibson --bddl --teleop --dataset
```

**Windows:**

```powershell
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

To skip the confirmation prompt when not in a conda environment:

```bash
./setup.sh --omnigibson --bddl --teleop --dataset --confirm-no-conda
```

### Automated/CI Installation & License Acceptance

For automated environments, you can automatically accept all required licenses:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

### Getting Help

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