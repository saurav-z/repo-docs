# BEHAVIOR-1K: Embodied AI for Everyday Tasks

**Tackle 1,000 real-world household activities with BEHAVIOR-1K, a comprehensive simulation benchmark for embodied AI research.** [Explore the original repository](https://github.com/StanfordVL/BEHAVIOR-1K).

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://behavior.stanford.edu/)

BEHAVIOR-1K provides a rich environment for training and evaluating embodied AI agents on a diverse set of human-centered tasks, from cooking and cleaning to organizing, using activities selected from human time-use surveys and preference studies.

**Key Features:**

*   **Extensive Task Coverage:**  1,000 diverse household activities.
*   **Realistic Simulation:** Built upon the OmniGibson physics simulator.
*   **Human-Centered:** Tasks derived from real-world human activities.
*   **Modular Installation:**  Install only the components you need.
*   **Teleoperation Support:**  JoyLo interface for robot teleoperation.

***For detailed information, visit our [main website](https://behavior.stanford.edu/)!***

## Installation Guide

Get started with BEHAVIOR-1K using the provided installation script.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ (recommended)
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

We recommend a full installation for most users. You can either create a new Conda environment or use your existing Python environment.

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

> **Important:** Run PowerShell as Administrator and set execution policy if necessary: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Installation Options and Components

Customize your installation with the following components:

### Available Components

| Component         | Flag             | Description                                                        |
| ----------------- | ---------------- | ------------------------------------------------------------------ |
| **OmniGibson**    | `--omnigibson`   | Core physics simulator and robotics environment                    |
| **BDDL**          | `--bddl`         | Behavior Domain Definition Language for task specification          |
| **Teleoperation** | `--teleop`       | JoyLo interface for robot teleoperation                           |

### Additional Options

| Option                      | Flag                     | Description                                                                                               |
| --------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------- |
| **New Environment**         | `--new-env`              | Create a new conda environment named `behavior` (requires conda)                                           |
| **Datasets**                | `--dataset`              | Download BEHAVIOR datasets (requires `--omnigibson`)                                                       |
| **Primitives**              | `--primitives`           | Install OmniGibson with action primitives support                                                        |
| **Development**             | `--dev`                  | Install development dependencies                                                                            |
| **CUDA Version**            | `--cuda-version X.X`     | Specify CUDA version (default: 12.4)                                                                       |
| **No Conda Confirmation**   | `--confirm-no-conda`     | Skip confirmation prompt when not in a conda environment                                                 |
| **Conda TOS**               | `--accept-conda-tos`     | Automatically accept Anaconda Terms of Service                                                             |
| **NVIDIA EULA**             | `--accept-nvidia-eula`   | Automatically accept NVIDIA Isaac Sim End User License Agreement                                           |
| **Dataset License**         | `--accept-dataset-tos`   | Automatically accept BEHAVIOR Data Bundle License Agreement                                             |

### Installation without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --teleop --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

To skip the confirmation prompt (useful for CI/CD):

```bash
./setup.sh --omnigibson --bddl --teleop --dataset --confirm-no-conda
```

To automatically accept all the terms of service and licenses, use these flags:
```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```
To view all available installation options:
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