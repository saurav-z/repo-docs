# BEHAVIOR-1K: Embodied AI for Everyday Tasks

**Tackle real-world challenges in embodied AI with BEHAVIOR-1K, a comprehensive benchmark featuring 1,000 household activities and realistic simulation.**

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

BEHAVIOR-1K is a cutting-edge simulation benchmark designed to evaluate and advance embodied AI agents. This project provides a complete toolkit for training and assessing agents on a diverse range of human-centered tasks, including cleaning, cooking, and organization. These activities are meticulously selected based on real-world human time-use surveys and preference studies, making BEHAVIOR-1K a valuable resource for developing AI that can seamlessly interact with and assist humans in their daily lives.

**Explore the full potential of BEHAVIOR-1K on our [main website](https://behavior.stanford.edu/).**

## Key Features

*   **1,000 Everyday Activities:** A vast and diverse set of tasks mirroring real-world human behaviors.
*   **Realistic Simulation:** Powered by the robust OmniGibson simulator.
*   **Human-Centered Design:** Tasks derived from real-world data on human activity.
*   **Modular Installation:** Choose the components you need for a customized setup.
*   **Comprehensive Tooling:** Provides everything you need to train and evaluate your agents.

## Installation Guide

BEHAVIOR-1K offers a streamlined installation process with a setup script. The script supports modular installation, allowing you to select only the necessary components.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

For the most straightforward setup, we recommend a full installation. You can either create a new Conda environment or use your existing Python environment.

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

> **Note:** Run PowerShell as Administrator and set the execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

#### Available Components

| Component          | Flag            | Description                                                            |
| ------------------ | --------------- | ---------------------------------------------------------------------- |
| OmniGibson         | `--omnigibson`  | Core physics simulator and robotics environment                        |
| BDDL               | `--bddl`        | Behavior Domain Definition Language for task specification               |
| Teleoperation      | `--teleop`      | JoyLo interface for robot teleoperation                                 |

#### Additional Options

| Option                  | Flag                      | Description                                                              |
| ----------------------- | ------------------------- | ------------------------------------------------------------------------ |
| New Environment         | `--new-env`               | Create a new conda environment named `behavior` (requires conda)          |
| Datasets                | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                    |
| Primitives              | `--primitives`            | Install OmniGibson with action primitives support                      |
| Development             | `--dev`                   | Install development dependencies                                         |
| CUDA Version            | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                     |
| No Conda Confirmation   | `--confirm-no-conda`      | Skip confirmation prompt when not in a conda environment                   |
| Accept Conda TOS       | `--accept-conda-tos`      | Automatically accept Anaconda Terms of Service                          |
| Accept NVIDIA EULA     | `--accept-nvidia-eula`    | Automatically accept NVIDIA Isaac Sim End User License Agreement          |
| Accept Dataset TOS       | `--accept-dataset-tos`    | Automatically accept BEHAVIOR Data Bundle License Agreement             |

### Installation without Conda

If you prefer to use your existing Python environment (system Python, venv, etc.), omit the `--new-env` flag:

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

To bypass all prompts for automated/CI environments, use the acceptance flags:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To view all available options, run:

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