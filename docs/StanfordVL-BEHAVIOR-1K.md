# BEHAVIOR-1K: Embodied AI for Everyday Tasks

**Tackle the complexities of real-world household activities with BEHAVIOR-1K, a groundbreaking simulation benchmark for training and evaluating embodied AI agents.** ([Original Repo](https://github.com/StanfordVL/BEHAVIOR-1K))

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://behavior.stanford.edu/)

BEHAVIOR-1K is a comprehensive simulation benchmark designed to push the boundaries of embodied AI.  It provides everything needed to train and evaluate agents on 1,000 realistic, human-centered activities, drawn from real-world data.

**Key Features:**

*   **Extensive Task Coverage:**  Simulates 1,000 everyday household activities, encompassing tasks like cleaning, cooking, and organizing.
*   **Realistic Simulation:** Leverages the OmniGibson physics simulator for accurate and detailed environment interactions.
*   **Human-Centered Design:** Activities are based on real-world human time-use surveys and preference studies.
*   **Modular Installation:** Install only the components you need.
*   **Teleoperation Support:** Includes a JoyLo interface for robot teleoperation.

***For more detailed information, please visit our [main website](https://behavior.stanford.edu/)!***

## Installation

This section provides instructions on how to install and set up BEHAVIOR-1K. The provided installation script simplifies the process, handling dependencies and allowing modular installation.

### System Requirements

Before you begin, ensure your system meets these requirements:

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

The following commands will perform a full installation, which is recommended for most users:

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

> **Important:**  On Windows, run PowerShell as Administrator and set the execution policy if necessary: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Installation Options

Customize your installation by selecting the desired components and options.

### Available Components

| Component          | Flag             | Description                                                      |
| ------------------ | ---------------- | ---------------------------------------------------------------- |
| **OmniGibson**     | `--omnigibson`   | Core physics simulator and robotics environment                  |
| **BDDL**           | `--bddl`         | Behavior Domain Definition Language for task specification          |
| **Teleoperation**  | `--teleop`       | JoyLo interface for robot teleoperation                          |

### Additional Options

| Option                      | Flag                      | Description                                                                                      |
| --------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------ |
| **New Environment**         | `--new-env`               | Create a new conda environment named `behavior` (requires conda)                                 |
| **Datasets**                | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                                            |
| **Primitives**              | `--primitives`            | Install OmniGibson with action primitives support                                             |
| **Development**             | `--dev`                   | Install development dependencies                                                                  |
| **CUDA Version**            | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                                             |
| **No Conda Confirmation** | `--confirm-no-conda`      | Skip confirmation prompt when not in a conda environment                                            |
| **Conda TOS**               | `--accept-conda-tos`      | Automatically accept Anaconda Terms of Service                                                   |
| **NVIDIA EULA**             | `--accept-nvidia-eula`    | Automatically accept NVIDIA Isaac Sim End User License Agreement                                 |
| **Dataset License**         | `--accept-dataset-tos`    | Automatically accept BEHAVIOR Data Bundle License Agreement                                      |

### Installation without Conda

If you prefer to use an existing Python environment (e.g., system Python, venv), omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --teleop --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

### Skipping Confirmation Prompts

To bypass confirmation prompts (useful for CI/CD), use the `--confirm-no-conda` flag:

```bash
./setup.sh --omnigibson --bddl --teleop --dataset --confirm-no-conda
```

### Automated Installation

For automated or CI/CD environments, use these flags to accept all necessary terms:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

### Getting Help

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