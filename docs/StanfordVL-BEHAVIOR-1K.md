# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark for Household Tasks

[![BEHAVIOR-1K](docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K** is a comprehensive benchmark designed to revolutionize embodied AI research, providing a realistic simulation environment for training and evaluating agents on 1,000 everyday household activities. Learn more on the [original GitHub repository](https://github.com/StanfordVL/BEHAVIOR-1K).

**Key Features:**

*   **1,000 Everyday Activities:** Test your AI agents on a wide range of human-centered tasks like cleaning, cooking, and organizing, drawn from real-world time-use surveys and preference studies.
*   **Realistic Simulation:** Built on top of the OmniGibson simulator, BEHAVIOR-1K offers a physically accurate and interactive environment.
*   **Modular Installation:** Easily install only the components you need.
*   **BDDL for Task Specification:** Leverage the Behavior Domain Definition Language (BDDL) for flexible task design and customization.
*   **JoyLo Interface:** Utilize the JoyLo interface for robot teleoperation and control.
*   **Extensive Dataset:** Access a rich dataset of tasks and environments to train and evaluate your AI agents.

## 🚀 Quick Start Installation

Get started with BEHAVIOR-1K quickly by following the simple steps below:

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Linux

```bash
# Clone the latest stable release (recommended)
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

### Windows

```powershell
# Clone the latest stable release (recommended)
git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

**Note:** If you prefer the latest development features (potentially less stable), clone the `main` branch instead of the `v3.7.1` tag.

### Advanced Installation Options

#### Available Components

| Component     | Flag            | Description                                       |
| ------------- | --------------- | ------------------------------------------------- |
| OmniGibson    | `--omnigibson`  | Core physics simulator and robotics environment   |
| BDDL          | `--bddl`        | Behavior Domain Definition Language              |
| JoyLo         | `--joylo`       | JoyLo interface for robot teleoperation           |

#### Additional Options

| Option                       | Flag                       | Description                                                                                                  |
| ---------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| New Environment              | `--new-env`                | Create a new conda environment named `behavior` (requires conda)                                            |
| Datasets                     | `--dataset`                | Download BEHAVIOR datasets (requires `--omnigibson`)                                                      |
| Primitives                   | `--primitives`             | Install OmniGibson with action primitives support                                                           |
| Eval                         | `--eval`                   | Install evaluation support for OmniGibson                                                                 |
| Development                  | `--dev`                    | Install development dependencies                                                                           |
| CUDA Version                 | `--cuda-version X.X`       | Specify CUDA version (default: 12.4)                                                                        |
| No Conda Confirmation        | `--confirm-no-conda`       | Skip confirmation prompt when not in a conda environment                                                  |
| Accept Conda TOS             | `--accept-conda-tos`       | Automatically accept Anaconda Terms of Service                                                              |
| Accept NVIDIA EULA           | `--accept-nvidia-eula`     | Automatically accept NVIDIA Isaac Sim End User License Agreement                                           |
| Accept Dataset License       | `--accept-dataset-tos`     | Automatically accept BEHAVIOR Data Bundle License Agreement                                                  |

### Installation Without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag.  For automated installs, accept required licenses.

```bash
# Linux (no conda)
./setup.sh --omnigibson --bddl --joylo --dataset --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos

# Windows (no conda)
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

To see all available options:
```bash
./setup.sh --help
```

## 📚 Citation

If you use BEHAVIOR-1K in your research, please cite the following paper:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```