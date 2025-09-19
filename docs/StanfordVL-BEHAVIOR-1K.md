# BEHAVIOR-1K: Embodied AI for Everyday Activities

![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)

**BEHAVIOR-1K is revolutionizing embodied AI by providing a comprehensive simulation benchmark for training and evaluating agents on 1,000 everyday household activities.** This open-source repository offers everything you need to get started with human-centered tasks like cleaning, cooking, and organizing, based on real-world human activity data. Explore the possibilities and push the boundaries of AI with BEHAVIOR-1K. [View the original repository on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K).

**Key Features:**

*   **Extensive Activity Coverage:** Train and evaluate agents on a diverse set of 1,000 common household activities.
*   **Realistic Simulation:** Utilizes advanced simulation environments for accurate and engaging agent training.
*   **Human-Centered Tasks:** Focuses on activities relevant to human time-use and preferences, ensuring practical application.
*   **Modular Installation:** Install only the necessary components for a streamlined setup.
*   **Open Source:** Leverage a community-driven project and contribute to the advancement of embodied AI.

**Learn More:** Check out our [main website](https://behavior.stanford.edu/) for in-depth information.

## 🚀 Installation

This repository provides an easy-to-use setup script that handles all dependencies and components. You can customize your installation to include only the features you require.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

We recommend installing the latest stable release (v3.7.1) with all components for most users:

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

> **Development Branch:** For the newest, potentially less stable features, clone the `main` branch:
> ```bash
> git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
> ```

> **Note for Windows:** Run PowerShell as Administrator and set execution policy if necessary: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## ⚙️ Installation Options

### Available Components

| Component         | Flag             | Description                                                                  |
| ----------------- | ---------------- | ---------------------------------------------------------------------------- |
| **OmniGibson**    | `--omnigibson`   | Core physics simulator and robotics environment                              |
| **BDDL**          | `--bddl`         | Behavior Domain Definition Language for task specification                   |
| **JoyLo**         | `--joylo`        | JoyLo interface for robot teleoperation                                     |

### Additional Options

| Option                      | Flag                          | Description                                                                            |
| --------------------------- | ----------------------------- | -------------------------------------------------------------------------------------- |
| **New Environment**         | `--new-env`                   | Create a new conda environment named `behavior` (requires conda)                         |
| **Datasets**                | `--dataset`                   | Download BEHAVIOR datasets (requires `--omnigibson`)                                     |
| **Primitives**              | `--primitives`                | Install OmniGibson with action primitives support                                      |
| **Eval**                    | `--eval`                      | Install evaluation support for OmniGibson                                              |
| **Development**             | `--dev`                       | Install development dependencies                                                        |
| **CUDA Version**            | `--cuda-version X.X`          | Specify CUDA version (default: 12.4)                                                   |
| **No Conda Confirmation**   | `--confirm-no-conda`          | Skip confirmation prompt when not in a conda environment                                |
| **Conda TOS**               | `--accept-conda-tos`          | Automatically accept Anaconda Terms of Service (for automated/CI environments)           |
| **NVIDIA EULA**             | `--accept-nvidia-eula`        | Automatically accept NVIDIA Isaac Sim End User License Agreement (for automated/CI env.) |
| **Dataset License**         | `--accept-dataset-tos`        | Automatically accept BEHAVIOR Data Bundle License Agreement (for automated/CI env.)      |

### Installation without Conda

If you prefer to use your existing Python environment (e.g., system Python, venv), omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --joylo --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

To skip the confirmation prompt when not using conda:

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
```

For automated/CI environments, you can bypass all prompts:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To view all available options:

```bash
./setup.sh --help
```

## 📚 Citation

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```