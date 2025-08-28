# BEHAVIOR-1K: Your Gateway to Human-Centered Embodied AI

[![BEHAVIOR-1K](docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K is a comprehensive simulation benchmark for training and evaluating embodied AI agents on a vast range of everyday household tasks.** This repository provides everything you need to immerse your agents in realistic scenarios, covering activities meticulously selected from real-world human time-use surveys and preference studies.

**[Visit our main website for more details!](https://behavior.stanford.edu/)**

## Key Features:

*   **1,000 Everyday Activities:** Test your agents on a diverse set of tasks, from cleaning and cooking to organizing and more.
*   **Realistic Simulation:** Leveraging advanced simulation technology, agents experience environments reflecting real-world physics and interactions.
*   **Human-Centered Design:** Tasks are grounded in human behavior and preferences, ensuring practical and relatable evaluation metrics.
*   **Comprehensive Toolkit:** Provides a monolithic repository with all necessary components for training and evaluating agents.
*   **Modular Installation:** Easily install only the required components for streamlined setup.

## Installation

Get started with BEHAVIOR-1K by following the straightforward installation process.

### System Requirements

*   **Operating System:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

We recommend a full installation for most users. Choose between creating a new Conda environment or using your existing Python environment:

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

> **Note:** Run PowerShell as Administrator and set the execution policy if required: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation with the following components and options:

**Available Components:**

| Component       | Flag              | Description                                          |
| --------------- | ----------------- | ---------------------------------------------------- |
| OmniGibson      | `--omnigibson`    | Core physics simulator and robotics environment      |
| BDDL            | `--bddl`          | Behavior Domain Definition Language for task specification |
| Teleoperation   | `--teleop`        | JoyLo interface for robot teleoperation              |

**Additional Options:**

| Option                   | Flag                      | Description                                                                                |
| ------------------------ | ------------------------- | ------------------------------------------------------------------------------------------ |
| New Environment          | `--new-env`               | Create a new conda environment named `behavior` (requires conda)                           |
| Datasets                 | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                                      |
| Primitives               | `--primitives`            | Install OmniGibson with action primitives support                                        |
| Development              | `--dev`                   | Install development dependencies                                                            |
| CUDA Version             | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                                        |
| No Conda Confirmation    | `--confirm-no-conda`      | Skip confirmation prompt when not in a conda environment                                    |
| Conda TOS                | `--accept-conda-tos`      | Automatically accept Anaconda Terms of Service                                            |
| NVIDIA EULA              | `--accept-nvidia-eula`    | Automatically accept NVIDIA Isaac Sim End User License Agreement                              |
| Dataset License          | `--accept-dataset-tos`    | Automatically accept BEHAVIOR Data Bundle License Agreement                                |

### Installation without Conda

Use your existing Python environment by omitting the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --teleop --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

Skip confirmation prompts (useful for CI/CD):

```bash
./setup.sh --omnigibson --bddl --teleop --dataset --confirm-no-conda
```

### Automated/CI Environments

Bypass all prompts for automated installation:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

For a full list of available options:
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