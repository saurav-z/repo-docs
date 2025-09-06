<h1 align="center">BEHAVIOR-1K: Embodied AI Benchmark for Everyday Activities</h1>

![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)

**Tackle the complexities of real-world human tasks with BEHAVIOR-1K, a comprehensive simulation benchmark designed to train and evaluate embodied AI agents on 1,000 household activities.** This repository provides everything you need to get started, including detailed task specifications and realistic simulated environments.  Explore the original repository on GitHub: [https://github.com/StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)

**Key Features:**

*   **1,000 Everyday Activities:** Train agents on a vast range of tasks like cleaning, cooking, and organizing, mirroring real human behavior.
*   **Realistic Simulation:** Leverages advanced simulation environments to provide a lifelike training ground for your AI agents.
*   **Modular Installation:** Install only the components you need, making setup flexible and efficient.
*   **Human-Centered Design:** Tasks are selected from real human time-use surveys and preference studies.
*   **Comprehensive:** Includes everything necessary for training, evaluation, and deployment of embodied AI agents.

## 🚀 Getting Started

### System Requirements

*   **Operating System:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Installation

BEHAVIOR-1K provides an easy-to-use setup script to handle all dependencies.

**Recommended: Install the latest stable release (v3.7.0) with all components:**

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

**For the latest development features (potentially less stable):**

```bash
# Clone the main branch
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

**Important Notes:**

*   **Windows:** Run PowerShell as Administrator and set the execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
*   **Conda Environment:** The `--new-env` flag creates a new Conda environment named `behavior`. If you prefer to use an existing environment, omit this flag.

### Installation Options

Customize your installation with the following flags:

| Component          | Flag            | Description                                                           |
|--------------------|-----------------|-----------------------------------------------------------------------|
| **OmniGibson**     | `--omnigibson`  | Core physics simulator and robotics environment                      |
| **BDDL**           | `--bddl`        | Behavior Domain Definition Language for task specification           |
| **JoyLo**          | `--joylo`       | JoyLo interface for robot teleoperation                             |
| **Datasets**       | `--dataset`     | Download BEHAVIOR datasets (requires `--omnigibson`)                |
| **Primitives**     | `--primitives`  | Install OmniGibson with action primitives support                    |
| **Eval**           | `--eval`        | Install evaluation support for OmniGibson                           |
| **Development**    | `--dev`         | Install development dependencies                                   |
| **CUDA Version**   | `--cuda-version X.X` | Specify CUDA version (default: 12.4)                                |
| **New Environment**| `--new-env`      | Create a new conda environment named `behavior` (requires conda)       |
| **No Conda Confirmation** | `--confirm-no-conda`      | Skip confirmation prompt when not in a conda environment       |

### Automated Installation & Terms of Service

For automated installations (e.g., CI/CD), use these flags to accept necessary terms:

| Option                 | Flag                         | Description                                                                      |
|------------------------|------------------------------|----------------------------------------------------------------------------------|
| **Conda TOS**          | `--accept-conda-tos`         | Automatically accept Anaconda Terms of Service                                   |
| **NVIDIA EULA**        | `--accept-nvidia-eula`       | Automatically accept NVIDIA Isaac Sim End User License Agreement                   |
| **Dataset License**    | `--accept-dataset-tos`       | Automatically accept BEHAVIOR Data Bundle License Agreement                       |

**Example for automated/CI environments:**

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To see all available options:
```bash
./setup.sh --help
```

## 📚 Citation

If you use BEHAVIOR-1K in your research, please cite the following:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```