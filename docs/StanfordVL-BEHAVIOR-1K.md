# BEHAVIOR-1K: A Benchmark for Embodied AI in Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**Challenge embodied AI with BEHAVIOR-1K, a comprehensive simulation benchmark featuring 1,000 realistic household tasks.**

This repository provides everything you need to train and evaluate embodied AI agents on human-centered activities like cleaning, cooking, and organizing, drawing from real-world time-use surveys and preference studies.  Explore the capabilities of your AI agents in a diverse and challenging environment.  Find more details on the [main website](https://behavior.stanford.edu/).

## Key Features

*   **Comprehensive Benchmark:** Evaluate agents on 1,000 diverse household activities.
*   **Realistic Simulation:** Powered by OmniGibson, offering high-fidelity physics and realistic environments.
*   **Human-Centered Tasks:** Activities selected from real-world human data.
*   **Modular Installation:** Install only the components you need.
*   **Flexible Setup:** Supports both Linux and Windows, with and without Conda.
*   **Detailed Documentation:** Comprehensive guides and options for various installation scenarios.

## Installation

Get started with BEHAVIOR-1K by following these steps:

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start - Recommended for most users:

Choose the appropriate instructions based on your OS. These commands will clone the latest stable release (v3.7.1) and install all components.

#### Linux

```bash
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

#### Windows (PowerShell as Administrator)

```powershell
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser # if needed
```

### Installation Options

Customize your installation with these components and options:

#### Available Components:

*   **OmniGibson:** (`--omnigibson`) Core physics simulator and robotics environment.
*   **BDDL:** (`--bddl`) Behavior Domain Definition Language for task specification.
*   **JoyLo:** (`--joylo`) JoyLo interface for robot teleoperation.

#### Additional Options:

*   **New Environment:** (`--new-env`) Creates a new conda environment named `behavior` (requires conda).
*   **Datasets:** (`--dataset`) Downloads BEHAVIOR datasets (requires `--omnigibson`).
*   **Primitives:** (`--primitives`) Installs OmniGibson with action primitives support.
*   **Eval:** (`--eval`) Installs evaluation support for OmniGibson.
*   **Development:** (`--dev`) Installs development dependencies.
*   **CUDA Version:** (`--cuda-version X.X`) Specifies CUDA version (default: 12.4).
*   **No Conda Confirmation:** (`--confirm-no-conda`) Skips confirmation prompt when not in a conda environment.
*   **Accept Terms of Service (for automated installs):**
    *   `--accept-conda-tos`: Automatically accepts Anaconda Terms of Service.
    *   `--accept-nvidia-eula`: Automatically accepts NVIDIA Isaac Sim EULA.
    *   `--accept-dataset-tos`: Automatically accepts BEHAVIOR Data Bundle License Agreement.

### Installation without Conda

If you're using an existing Python environment, omit the `--new-env` flag:

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

For a full list of options run `./setup.sh --help`.

### Development Branch

For the latest features (potentially less stable), clone the main branch:

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
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

---

**[Back to Top](https://github.com/StanfordVL/BEHAVIOR-1K)**