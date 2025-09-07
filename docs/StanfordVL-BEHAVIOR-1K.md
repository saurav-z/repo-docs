# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark for Everyday Tasks

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K offers a comprehensive simulation environment to train and evaluate embodied AI agents on 1,000 realistic household activities.** This powerful benchmark, developed by Stanford Vision and Learning Group, provides everything needed for researchers and developers to push the boundaries of embodied AI in human-centered tasks like cleaning, cooking, and organizing.  Learn more on the [main website](https://behavior.stanford.edu/) or dive directly into the code on [GitHub](https://github.com/StanfordVL/BEHAVIOR-1K).

## Key Features

*   **1,000 Everyday Activities:** Test your agents on a diverse set of tasks drawn from real-world human time-use surveys and preference studies.
*   **Realistic Simulation:** Leverage the power of the OmniGibson physics simulator for a truly immersive experience.
*   **Modular Installation:** Easily install only the components you need.
*   **Comprehensive Tooling:** Includes Behavior Domain Definition Language (BDDL) for task specification and JoyLo interface for robot teleoperation.
*   **Open Source:** Fully accessible and available for research and development.

## Installation Guide

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

Follow these steps to get started with the latest stable release.

**Linux:**

```bash
# Clone the latest stable release (recommended)
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

**Windows:**

```powershell
# Clone the latest stable release (recommended)
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

**Note:** Run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

#### Development Branch

For the latest features (potentially less stable), clone the `main` branch:

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

### Installation Options

Customize your installation with the following components and options.

#### Available Components

| Component      | Flag          | Description                                              |
| -------------- | ------------- | -------------------------------------------------------- |
| **OmniGibson** | `--omnigibson` | Core physics simulator and robotics environment          |
| **BDDL**       | `--bddl`      | Behavior Domain Definition Language for task specification |
| **JoyLo**      | `--joylo`     | JoyLo interface for robot teleoperation                 |

#### Additional Options

| Option                   | Flag                      | Description                                                                         |
| ------------------------ | ------------------------- | ----------------------------------------------------------------------------------- |
| **New Environment**      | `--new-env`               | Create a new conda environment named `behavior` (requires conda)                    |
| **Datasets**             | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                                |
| **Primitives**           | `--primitives`            | Install OmniGibson with action primitives support                                  |
| **Eval**                 | `--eval`                  | Install evaluation support for OmniGibson                                          |
| **Development**          | `--dev`                   | Install development dependencies                                                   |
| **CUDA Version**         | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                                  |
| **No Conda Confirmation** | `--confirm-no-conda`      | Skip confirmation prompt when not in a conda environment                            |
| **Conda TOS**            | `--accept-conda-tos`      | Automatically accept Anaconda Terms of Service                                      |
| **NVIDIA EULA**          | `--accept-nvidia-eula`    | Automatically accept NVIDIA Isaac Sim End User License Agreement                      |
| **Dataset License**      | `--accept-dataset-tos`    | Automatically accept BEHAVIOR Data Bundle License Agreement                           |

### Installation without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --joylo --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

To skip the confirmation prompt if not in a conda environment:

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
```

### Terms of Service & License Acceptance

For non-interactive installations, use the following flags to accept terms:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To view all available options:
```bash
./setup.sh --help
```

## 📄 Citation

If you use BEHAVIOR-1K in your research, please cite the following paper:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```