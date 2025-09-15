# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark for Household Tasks

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**Tackle the complexity of real-world tasks with BEHAVIOR-1K, a cutting-edge simulation benchmark for embodied AI agents, focusing on 1,000 everyday household activities.**  This comprehensive repository provides all the tools needed to train and evaluate agents on tasks like cooking, cleaning, and organizing, derived from real-world human behavior data.

***For more details, visit our [main website](https://behavior.stanford.edu/)!***

## Key Features:

*   **1,000 Everyday Activities:**  Test your AI agents on a vast and diverse range of household tasks.
*   **Human-Centered Tasks:**  Focuses on activities derived from real human time-use surveys and preference studies.
*   **Realistic Simulation:** Built using OmniGibson, providing a robust and interactive environment.
*   **Modular Installation:** Easily install only the components you need.
*   **Comprehensive Benchmark:**  Includes everything needed for training and evaluating embodied AI agents.

## Getting Started

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ (recommended)
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Installation Instructions

#### Recommended: Install the Latest Stable Release (v3.7.1)

Follow these steps for a quick and easy setup:

##### Linux

```bash
# Clone the latest stable release
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

##### Windows

```powershell
# Clone the latest stable release
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

**Note:** On Windows, run PowerShell as Administrator and set the execution policy if required: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

#### Alternative: Install the Development Branch

For the latest, but potentially less stable, features:

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

### Installation Options and Components

You can customize your installation with various components and options.

#### Available Components

| Component       | Flag             | Description                                                              |
| --------------- | ---------------- | ------------------------------------------------------------------------ |
| OmniGibson      | `--omnigibson`   | Core physics simulator and robotics environment                          |
| BDDL            | `--bddl`         | Behavior Domain Definition Language for task specification               |
| JoyLo           | `--joylo`        | JoyLo interface for robot teleoperation                                 |

#### Additional Options

| Option                 | Flag                    | Description                                                                            |
| ---------------------- | ----------------------- | -------------------------------------------------------------------------------------- |
| New Environment        | `--new-env`             | Create a new conda environment named `behavior` (requires conda)                         |
| Datasets               | `--dataset`             | Download BEHAVIOR datasets (requires `--omnigibson`)                                      |
| Primitives             | `--primitives`          | Install OmniGibson with action primitives support                                      |
| Evaluation             | `--eval`                | Install evaluation support for OmniGibson                                                |
| Development            | `--dev`                 | Install development dependencies                                                         |
| CUDA Version           | `--cuda-version X.X`    | Specify CUDA version (default: 12.4)                                                   |
| No Conda Confirmation | `--confirm-no-conda`    | Skip confirmation prompt when not in a conda environment                                   |
| Conda TOS              | `--accept-conda-tos`    | Automatically accept Anaconda Terms of Service                                       |
| NVIDIA EULA            | `--accept-nvidia-eula`  | Automatically accept NVIDIA Isaac Sim End User License Agreement                         |
| Dataset License        | `--accept-dataset-tos`  | Automatically accept BEHAVIOR Data Bundle License Agreement                            |

### Installation Without Conda

If you prefer to use your existing Python environment, simply omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --joylo --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

To skip the confirmation prompt when not in a conda environment:

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
```

### Accepting Terms of Service and Licenses

For automated or CI environments, use the following flags to accept the necessary terms and licenses:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

### Further Help

To view all available options, run:

```bash
./setup.sh --help
```

## Citation

If you use BEHAVIOR-1K in your research, please cite the following:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```

---

**[Visit the original repository on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K) for more information and to contribute!**