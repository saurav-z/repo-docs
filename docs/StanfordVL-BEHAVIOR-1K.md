# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark for Human-Centered Tasks

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K** is a comprehensive simulation benchmark designed to challenge and advance embodied AI by testing agents on 1,000 realistic household activities. This repository provides everything you need to get started, including detailed instructions and installation scripts.  Explore the full capabilities of BEHAVIOR-1K on the [main website](https://behavior.stanford.edu/).

## Key Features

*   **1,000 Everyday Activities:**  Focuses on tasks derived from real-world human time-use surveys and preference studies, covering cleaning, cooking, organizing, and more.
*   **Realistic Simulation:**  Built upon the robust OmniGibson physics simulator for accurate and reliable agent interactions.
*   **Modular Installation:**  Offers flexible installation options to tailor the setup to your specific needs.
*   **Teleoperation Interface:**  Includes a JoyLo interface for direct robot teleoperation.
*   **Behavior Domain Definition Language (BDDL):**  Leverages BDDL for precise task specification.

## Installation Guide

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

For the easiest setup, we recommend a full installation using the provided scripts.  You can either create a new conda environment or use your existing Python environment.

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

> **Note:**  On Windows, run PowerShell as Administrator and consider setting the execution policy if necessary: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation with these components:

| Component          | Flag            | Description                                    |
|--------------------|-----------------|------------------------------------------------|
| OmniGibson         | `--omnigibson`   | Core physics simulator and robotics environment |
| BDDL               | `--bddl`         | Behavior Domain Definition Language           |
| Teleoperation      | `--teleop`       | JoyLo interface for robot teleoperation        |

**Additional Options:**

| Option                    | Flag                      | Description                                                                 |
|---------------------------|---------------------------|-----------------------------------------------------------------------------|
| New Environment           | `--new-env`               | Create a new conda environment named `behavior` (requires conda)           |
| Datasets                  | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                         |
| Primitives                | `--primitives`            | Install OmniGibson with action primitives support                              |
| Development               | `--dev`                   | Install development dependencies                                            |
| CUDA Version              | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                        |
| No Conda Confirmation     | `--confirm-no-conda`      | Skip confirmation prompt when not in a conda environment                     |
| Conda TOS                 | `--accept-conda-tos`      | Automatically accept Anaconda Terms of Service                               |
| NVIDIA EULA               | `--accept-nvidia-eula`    | Automatically accept NVIDIA Isaac Sim End User License Agreement               |
| Dataset License           | `--accept-dataset-tos`    | Automatically accept BEHAVIOR Data Bundle License Agreement                |

### Installation without Conda

To use your existing Python environment, omit the `--new-env` flag:

**Linux:**

```bash
./setup.sh --omnigibson --bddl --teleop --dataset
```

**Windows:**

```powershell
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

To skip confirmation prompts when not in a conda environment (useful for CI/CD):

```bash
./setup.sh --omnigibson --bddl --teleop --dataset --confirm-no-conda
```

### Automated Installation (CI/CD)

For automated environments, accept all licenses and bypass all prompts:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

### Help

For a comprehensive list of installation options, use:

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