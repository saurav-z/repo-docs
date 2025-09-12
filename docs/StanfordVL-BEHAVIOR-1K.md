<h1 align="center">BEHAVIOR-1K: Embodied AI Benchmark</h1>

<p align="center">
  <img src="./docs/assets/readme_splash_logo.png" alt="BEHAVIOR-1K Logo" width="600"/>
</p>

**Tackle the complexities of everyday life with BEHAVIOR-1K, a comprehensive simulation benchmark designed to train and evaluate embodied AI agents on 1,000 household activities.**  This repository provides everything you need to build AI agents capable of performing realistic human-centered tasks like cleaning, cooking, and organization, based on real-world human activity data.  [Explore the original repository on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K).

## Key Features of BEHAVIOR-1K:

*   **Extensive Task Coverage:** Tests agents on 1,000 diverse household activities drawn from real-world human time-use surveys and preference studies.
*   **Realistic Simulation:** Utilizes the OmniGibson physics simulator to create realistic and interactive environments.
*   **Modular Installation:**  Provides a flexible installation script to easily install all necessary components, including the physics simulator, task specification language, and robot teleoperation interface.
*   **Human-Centered Design:** Focused on activities relevant to human daily life, facilitating the development of agents that can assist with everyday tasks.
*   **Reproducible Research:** Enables researchers to consistently evaluate and compare the performance of embodied AI agents.

## Installation Guide

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start (Recommended)

Install the latest stable release (v3.7.0) with all components:

#### Linux

```bash
# Clone the latest stable release (recommended)
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

#### Windows

```powershell
# Clone the latest stable release (recommended)
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

**Note:** For the latest development features (potentially less stable), clone the `main` branch.

### Installation Options

The `setup.sh` (Linux) and `setup.ps1` (Windows) scripts offer several options to customize the installation:

**Available Components:**

| Component        | Flag           | Description                                         |
| ---------------- | -------------- | --------------------------------------------------- |
| OmniGibson       | `--omnigibson` | Core physics simulator and robotics environment     |
| BDDL             | `--bddl`       | Behavior Domain Definition Language for task specification |
| JoyLo            | `--joylo`      | JoyLo interface for robot teleoperation               |

**Additional Options:**

| Option                      | Flag                       | Description                                                                   |
| --------------------------- | -------------------------- | ----------------------------------------------------------------------------- |
| New Environment             | `--new-env`                | Create a new conda environment named `behavior` (requires conda)              |
| Datasets                    | `--dataset`                | Download BEHAVIOR datasets (requires `--omnigibson`)                           |
| Primitives                  | `--primitives`             | Install OmniGibson with action primitives support                             |
| Eval                        | `--eval`                   | Install evaluation support for OmniGibson                                     |
| Development                 | `--dev`                    | Install development dependencies                                                |
| CUDA Version                | `--cuda-version X.X`       | Specify CUDA version (default: 12.4)                                          |
| No Conda Confirmation       | `--confirm-no-conda`       | Skip confirmation prompt when not in a conda environment                       |
| Accept Conda TOS            | `--accept-conda-tos`       | Automatically accept Anaconda Terms of Service                                 |
| Accept NVIDIA EULA          | `--accept-nvidia-eula`     | Automatically accept NVIDIA Isaac Sim End User License Agreement               |
| Accept Dataset License      | `--accept-dataset-tos`     | Automatically accept BEHAVIOR Data Bundle License Agreement                     |

### Installation without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --joylo --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

### Automated Installation (CI/CD)

For automated installations, use flags to bypass prompts:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

## Resources

*   **Main Website:** [https://behavior.stanford.edu/](https://behavior.stanford.edu/)
*   **Original Repository:** [https://github.com/StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
*   **To see all available options:** `./setup.sh --help`

## Citation

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```