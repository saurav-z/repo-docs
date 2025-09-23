# BEHAVIOR-1K: A Benchmark for Embodied AI in Everyday Life

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K** provides a comprehensive simulation environment to train and evaluate embodied AI agents on 1,000 realistic household activities, offering a vital tool for advancing AI in human-centered tasks.  Explore the original repository [here](https://github.com/StanfordVL/BEHAVIOR-1K).

**Key Features:**

*   **1,000 Everyday Activities:**  Tests AI agents on a vast range of tasks based on real-world human activity data (cleaning, cooking, organizing, etc.).
*   **Realistic Simulation:** Built on the OmniGibson physics simulator, offering a high-fidelity environment for agent training and evaluation.
*   **Human-Centered Tasks:**  Focuses on activities derived from human time-use surveys and preference studies, making it highly relevant to real-world applications.
*   **Modular Installation:**  Offers a flexible setup, allowing users to install only the necessary components for their specific projects.
*   **Comprehensive Support:** Includes BDDL for task specification and JoyLo for robot teleoperation, providing a complete development environment.

## Installation Guide

Get started with BEHAVIOR-1K by following the simple installation steps outlined below.  The installation script handles dependencies and simplifies the setup process.

### System Requirements

*   **Operating System:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start Installation (Recommended)

This guide installs the latest stable release (v3.7.1) with all components.

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

> **Development Branch:**  For the newest features (potentially less stable), clone the `main` branch instead:
> ```bash
> git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
> ```

> **Note:**  On Windows, run PowerShell as Administrator and set the execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation with the following components and options.

#### Available Components

| Component      | Flag           | Description                                                                 |
|----------------|----------------|-----------------------------------------------------------------------------|
| **OmniGibson** | `--omnigibson` | Core physics simulator and robotics environment                             |
| **BDDL**       | `--bddl`       | Behavior Domain Definition Language for task specification                     |
| **JoyLo**      | `--joylo`      | JoyLo interface for robot teleoperation                                      |

#### Additional Options

| Option                      | Flag                       | Description                                                                                     |
|-----------------------------|----------------------------|-------------------------------------------------------------------------------------------------|
| **New Environment**         | `--new-env`                | Create a new conda environment named `behavior` (requires conda)                               |
| **Datasets**                | `--dataset`                | Download BEHAVIOR datasets (requires `--omnigibson`)                                            |
| **Primitives**              | `--primitives`             | Install OmniGibson with action primitives support                                            |
| **Eval**                    | `--eval`                   | Install evaluation support for OmniGibson                                                      |
| **Development**             | `--dev`                    | Install development dependencies                                                              |
| **CUDA Version**            | `--cuda-version X.X`       | Specify CUDA version (default: 12.4)                                                            |
| **No Conda Confirmation**   | `--confirm-no-conda`       | Skip confirmation prompt when not in a conda environment                                        |
| **Conda TOS**               | `--accept-conda-tos`       | Automatically accept Anaconda Terms of Service                                              |
| **NVIDIA EULA**             | `--accept-nvidia-eula`     | Automatically accept NVIDIA Isaac Sim End User License Agreement                                |
| **Dataset License**         | `--accept-dataset-tos`     | Automatically accept BEHAVIOR Data Bundle License Agreement                                     |

### Installation without Conda

If you want to use your existing Python environment (system Python, venv, etc.) rather than conda, omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --joylo --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

If you are not in a conda environment, the script will ask for confirmation. Use the `--confirm-no-conda` to bypass it:

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
```

### Automated Installation & License Acceptance

For automated installations (e.g., in CI/CD environments), use the flags below to accept the necessary terms and licenses.

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

### For Help

To see all installation options and their descriptions:
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