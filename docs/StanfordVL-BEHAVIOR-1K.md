# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark for Everyday Tasks

![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)

**BEHAVIOR-1K provides a comprehensive simulation environment for training and evaluating embodied AI agents on 1,000 realistic household activities.**  This benchmark allows researchers to develop and test AI agents on human-centered tasks like cleaning, cooking, and organizing, drawing data from real-world human behavior and preferences. Explore the cutting edge of AI with BEHAVIOR-1K!

👉 [Explore the BEHAVIOR-1K Project on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K) and our [main website](https://behavior.stanford.edu/) for more details!

## Key Features

*   **1,000 Everyday Activities:**  Test your agents on a wide range of realistic household tasks.
*   **Human-Centered Tasks:** Focus on activities derived from real human time-use surveys and preference studies.
*   **Comprehensive Simulation:**  A complete environment for training, testing, and evaluating embodied AI agents.
*   **Modular Installation:** Install only the components you need.
*   **Realistic Environments:** Powered by OmniGibson for high-fidelity physics and robotics simulation.
*   **Flexible Task Specification:** Utilize Behavior Domain Definition Language (BDDL) for task definition.

## Installation

Get started with BEHAVIOR-1K by following these installation instructions.  The setup script handles dependencies and components, allowing for modular installations.

### System Requirements

*   **Operating System:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

We recommend installing the latest stable release (v3.7.0) with all components for most users.

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

> **Development Branch:**  For the latest features (potentially less stable), clone the `main` branch:
> ```bash
> git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
> ```
> **Note:**  Run PowerShell as Administrator and set the execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

#### Available Components

| Component      | Flag          | Description                                                |
|----------------|---------------|------------------------------------------------------------|
| OmniGibson     | `--omnigibson` | Core physics simulator and robotics environment             |
| BDDL           | `--bddl`      | Behavior Domain Definition Language for task specification |
| JoyLo          | `--joylo`     | JoyLo interface for robot teleoperation                    |

#### Additional Options

| Option                    | Flag                       | Description                                                                     |
|---------------------------|----------------------------|---------------------------------------------------------------------------------|
| New Environment           | `--new-env`                | Create a new conda environment named `behavior` (requires conda)               |
| Datasets                  | `--dataset`                | Download BEHAVIOR datasets (requires `--omnigibson`)                             |
| Primitives                | `--primitives`             | Install OmniGibson with action primitives support                               |
| Evaluation                | `--eval`                   | Install evaluation support for OmniGibson                                      |
| Development               | `--dev`                    | Install development dependencies                                               |
| CUDA Version              | `--cuda-version X.X`       | Specify CUDA version (default: 12.4)                                            |
| No Conda Confirmation     | `--confirm-no-conda`       | Skip confirmation prompt when not in a conda environment                        |
| Accept Conda TOS          | `--accept-conda-tos`       | Automatically accept Anaconda Terms of Service                                  |
| Accept NVIDIA EULA       | `--accept-nvidia-eula`     | Automatically accept NVIDIA Isaac Sim End User License Agreement                  |
| Accept Dataset License    | `--accept-dataset-tos`     | Automatically accept BEHAVIOR Data Bundle License Agreement                       |

### Installation without Conda

If you prefer using an existing Python environment, omit the `--new-env` flag.

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

For non-interactive/automated installations, use the following flags to accept necessary terms of service:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To view all available options:
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