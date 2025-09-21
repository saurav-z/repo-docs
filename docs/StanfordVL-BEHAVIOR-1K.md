# BEHAVIOR-1K: Embodied AI for Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**Tackle real-world tasks with BEHAVIOR-1K, a comprehensive benchmark for training and evaluating embodied AI agents on 1,000 everyday household activities.** This cutting-edge platform allows researchers to explore human-centered tasks such as cooking, cleaning, and organizing, all within a realistic simulated environment. For more details, visit our [main website](https://behavior.stanford.edu/).

**[View the original repository on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K)**

## Key Features

*   **Extensive Activity Suite:** Evaluates agents on 1,000 diverse household activities, derived from real-world human time-use surveys and preference studies.
*   **Realistic Simulation:** Leverages the OmniGibson physics simulator for accurate and interactive environment modeling.
*   **Modular Installation:** Offers flexible component selection, enabling users to install only the necessary modules.
*   **Human-Centered Design:** Focuses on tasks and scenarios that reflect everyday human experiences.
*   **Open-Source & Accessible:** Provides a readily available platform for researchers and developers to advance embodied AI research.

## Installation Guide

Get started with BEHAVIOR-1K using the provided setup script, which handles dependencies and component installation.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

We recommend installing the latest stable release for most users:

#### Linux

```bash
# Clone the latest stable release
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

#### Windows

```powershell
# Clone the latest stable release
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

**Note:** For the latest development features (potentially less stable), clone the `main` branch instead. Remember to run PowerShell as Administrator and set the execution policy if necessary: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

### Installation Options

Choose the components you need during installation:

#### Available Components

| Component       | Flag          | Description                                                  |
| --------------- | ------------- | ------------------------------------------------------------ |
| OmniGibson      | `--omnigibson` | Core physics simulator and robotics environment              |
| BDDL            | `--bddl`      | Behavior Domain Definition Language for task specification   |
| JoyLo           | `--joylo`     | JoyLo interface for robot teleoperation                     |
| **Datasets**    | `--dataset`   | Downloads the BEHAVIOR-1K datasets (requires `--omnigibson`) |

#### Additional Options

| Option                  | Flag                       | Description                                                                                                |
| ----------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------- |
| New Environment         | `--new-env`                | Creates a new conda environment named `behavior` (requires conda)                                          |
| Primitives              | `--primitives`             | Installs OmniGibson with action primitives support                                                       |
| Eval                    | `--eval`                   | Installs evaluation support for OmniGibson                                                               |
| Development             | `--dev`                    | Installs development dependencies                                                                           |
| CUDA Version            | `--cuda-version X.X`       | Specifies the CUDA version (default: 12.4)                                                               |
| No Conda Confirmation   | `--confirm-no-conda`       | Skips the confirmation prompt when not in a conda environment                                             |
| Accept Conda TOS        | `--accept-conda-tos`       | Automatically accepts Anaconda Terms of Service                                                           |
| Accept NVIDIA EULA      | `--accept-nvidia-eula`     | Automatically accepts NVIDIA Isaac Sim End User License Agreement                                        |
| Accept Dataset License  | `--accept-dataset-tos`     | Automatically accepts BEHAVIOR Data Bundle License Agreement                                              |

### Installation without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --joylo --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

To skip the confirmation prompt if you're not using conda:

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
```

For automated/CI environments, you can bypass all prompts with:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

For a complete list of installation options:
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