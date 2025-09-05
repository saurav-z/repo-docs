# BEHAVIOR-1K: Embodied AI for Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**Tackle the complexities of human-centered tasks with BEHAVIOR-1K, a comprehensive simulation benchmark designed to train and evaluate embodied AI agents.** This repository provides everything you need to get started, including a wide range of household activities drawn from real-world data.

🔗 **[View the original repository on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K)**

**Key Features:**

*   **1,000 Everyday Activities:** Explore a vast and diverse set of human-centered tasks, including cleaning, cooking, and organization.
*   **Realistic Simulation:** Built on the powerful OmniGibson physics simulator to provide a realistic and interactive environment.
*   **Modular Installation:** Easily install only the components you need with a flexible setup script.
*   **Human-Centered Design:** Activities are based on real human time-use surveys and preference studies.
*   **Easy Setup:** Get up and running quickly with straightforward installation instructions for Linux and Windows.

---

## 🛠️ Installation

The BEHAVIOR-1K project offers a streamlined installation process using a dedicated setup script. This script handles all dependencies and components, providing modular installation options for flexibility.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+ recommended
*   **GPU:** NVIDIA RTX 2080+ or better

### Quick Start

For most users, we recommend installing the latest stable release (v3.7.0) with all components:

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

> **Development Branch:** To access the latest features (potentially less stable), clone the `main` branch instead:
> ```bash
> git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
> ```

> **Note:** Run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

#### Available Components

| Component       | Flag          | Description                                                  |
|-----------------|---------------|--------------------------------------------------------------|
| **OmniGibson** | `--omnigibson` | Core physics simulator and robotics environment              |
| **BDDL**       | `--bddl`       | Behavior Domain Definition Language for task specification |
| **JoyLo**       | `--joylo`       | JoyLo interface for robot teleoperation                      |

#### Additional Options

| Option                      | Flag                        | Description                                                                  |
|-----------------------------|-----------------------------|------------------------------------------------------------------------------|
| **New Environment**         | `--new-env`                 | Create a new conda environment named `behavior` (requires conda)              |
| **Datasets**                | `--dataset`                 | Download BEHAVIOR datasets (requires `--omnigibson`)                          |
| **Primitives**              | `--primitives`              | Install OmniGibson with action primitives support                            |
| **Eval**                    | `--eval`                    | Install evaluation support for OmniGibson                                    |
| **Development**             | `--dev`                     | Install development dependencies                                              |
| **CUDA Version**            | `--cuda-version X.X`        | Specify CUDA version (default: 12.4)                                         |
| **No Conda Confirmation**   | `--confirm-no-conda`        | Skip confirmation prompt when not in a conda environment                       |
| **Accept Conda TOS**        | `--accept-conda-tos`        | Automatically accept Anaconda Terms of Service                               |
| **Accept NVIDIA EULA**      | `--accept-nvidia-eula`      | Automatically accept NVIDIA Isaac Sim End User License Agreement              |
| **Accept Dataset License**  | `--accept-dataset-tos`      | Automatically accept BEHAVIOR Data Bundle License Agreement                   |

### Installation without Conda

If you prefer to use your existing Python environment (system Python, venv, etc.) instead of conda, simply omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --joylo --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

To skip the confirmation prompt when not in a conda environment (useful for CI/CD):

```bash
./setup.sh --omnigibson --bddl --joylo --dataset --confirm-no-conda
```

### Terms of Service & License Acceptance

BEHAVIOR-1K installation requires accepting various terms of service and license agreements. Use the following flags for automated installations:

To see all available options:

```bash
./setup.sh --help
```

---

## 📄 Citation

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```