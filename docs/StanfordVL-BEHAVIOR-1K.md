# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K empowers researchers to train and evaluate embodied AI agents on a diverse set of 1,000 realistic, human-centered household tasks like cooking and cleaning.** This comprehensive simulation benchmark, built on real-world data, provides everything needed to advance the field of embodied AI.

👉 **Explore the full details and learn more on the [BEHAVIOR-1K website](https://behavior.stanford.edu/).**

## Key Features

*   **Extensive Task Suite:** Benchmark agents on 1,000 everyday household activities, mirroring real human time-use.
*   **Realistic Simulation:** Leverages advanced physics simulators for authentic agent interactions.
*   **Human-Centered Design:** Tasks are drawn from real-world surveys and preference studies.
*   **Modular Installation:** Install only the components you need.
*   **Comprehensive Documentation:**  Easy-to-follow instructions and options for flexible installation.

## Getting Started

### System Requirements

*   **Operating System:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Installation Guide

Follow these steps to install BEHAVIOR-1K. It is recommended to clone and install the latest stable release (v3.7.0):

#### Linux

```bash
# Clone the repository
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

#### Windows

```powershell
# Clone the repository
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

**Note:** If you wish to use the latest development features, clone the `main` branch instead.

### Installation Options

The setup script provides several options for a customizable installation.

**Available Components:**

*   `--omnigibson`: Core physics simulator and robotics environment.
*   `--bddl`: Behavior Domain Definition Language for task specification.
*   `--joylo`: JoyLo interface for robot teleoperation.

**Additional Options:**

*   `--new-env`: Creates a new conda environment named `behavior`.
*   `--dataset`: Downloads BEHAVIOR datasets (requires `--omnigibson`).
*   `--primitives`: Installs OmniGibson with action primitives support.
*   `--eval`: Installs evaluation support for OmniGibson.
*   `--dev`: Installs development dependencies.
*   `--cuda-version X.X`: Specifies the CUDA version (default: 12.4).
*   `--confirm-no-conda`: Skips confirmation prompt when not in a conda environment.
*   `--accept-conda-tos`, `--accept-nvidia-eula`, `--accept-dataset-tos`: Automatically accept terms of service and licenses for automated installations.

To view all available options:

```bash
./setup.sh --help
```

### Installation Without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag during installation:

#### Linux

```bash
./setup.sh --omnigibson --bddl --joylo --dataset
```

#### Windows

```powershell
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

If you're not in a conda environment, the script will prompt for confirmation. To skip this prompt:

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
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

For more information, please visit the [BEHAVIOR-1K GitHub Repository](https://github.com/StanfordVL/BEHAVIOR-1K).