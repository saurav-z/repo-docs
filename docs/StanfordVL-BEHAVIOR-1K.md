# BEHAVIOR-1K: A Benchmark for Embodied AI in Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K is a comprehensive simulation benchmark enabling researchers to train and evaluate embodied AI agents on 1,000 realistic household activities.**  This repository provides the tools and resources to tackle human-centered tasks like cleaning, cooking, and organizing, all drawn from real-world data.

**Explore the full potential of embodied AI.  [Visit the main website for more details](https://behavior.stanford.edu/)**

## Key Features:

*   **1,000 Everyday Activities:**  Focuses on human-centered tasks for realistic AI training.
*   **Comprehensive Simulation:** Integrates with OmniGibson for physics-based simulations.
*   **Modular Installation:** Easily install only the components you need.
*   **Real-World Data:**  Leverages data from human time-use surveys and preference studies.
*   **Open Source:**  Freely available for research and development.

## Installation Guide

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quickstart Installation

**Choose your operating system:**

#### Linux

```bash
# Clone the latest stable release (recommended)
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

#### Windows

```powershell
# Clone the latest stable release (recommended)
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

**Note**: Ensure PowerShell is run as Administrator and set execution policy (if needed):  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**For development features (potentially less stable):**
```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

### Installation Options and Customization

The `setup.sh` (Linux) and `setup.ps1` (Windows) scripts provide flexibility.  Use the following flags to customize your installation:

#### Available Components:

| Component      | Flag           | Description                                              |
|----------------|----------------|----------------------------------------------------------|
| OmniGibson     | `--omnigibson`   | Core physics simulator and robotics environment           |
| BDDL           | `--bddl`         | Behavior Domain Definition Language for task specification |
| JoyLo          | `--joylo`        | JoyLo interface for robot teleoperation                  |

#### Additional Options:

| Option                       | Flag                      | Description                                                                         |
|------------------------------|---------------------------|-------------------------------------------------------------------------------------|
| New Environment              | `--new-env`               | Create a new conda environment named `behavior` (requires conda)                  |
| Datasets                     | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                             |
| Primitives                   | `--primitives`            | Install OmniGibson with action primitives support                                 |
| Evaluation Support          | `--eval`                  | Install evaluation support for OmniGibson                                         |
| Development Dependencies     | `--dev`                   | Install development dependencies                                                    |
| CUDA Version                 | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                                 |
| Skip Conda Confirmation      | `--confirm-no-conda`      | Skip confirmation prompt when not in a conda environment                               |
| Accept Conda Terms of Service | `--accept-conda-tos`    | Automatically accept Anaconda Terms of Service                                      |
| Accept NVIDIA EULA           | `--accept-nvidia-eula`  | Automatically accept NVIDIA Isaac Sim End User License Agreement                    |
| Accept Dataset License       | `--accept-dataset-tos`  | Automatically accept BEHAVIOR Data Bundle License Agreement                          |

### Installation Without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --joylo --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

### Automated/CI Installation

For automated installations (CI/CD), bypass prompts:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

### Help

For a full list of installation options:

```bash
./setup.sh --help
```

## Citation

If you use BEHAVIOR-1K in your research, please cite:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}