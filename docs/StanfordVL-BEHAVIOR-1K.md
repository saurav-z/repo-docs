<h1 align="center">BEHAVIOR-1K: Embodied AI for Everyday Tasks</h1>

<p align="center">
  <img src="./docs/assets/readme_splash_logo.png" alt="BEHAVIOR-1K Logo">
</p>

**BEHAVIOR-1K empowers researchers to train and evaluate embodied AI agents in realistic household environments by offering a comprehensive benchmark of 1,000 diverse, human-centered activities.**  Explore the complexity of tasks like cooking, cleaning, and organizing, using a platform built for real-world relevance.  This repository provides all the necessary tools to get started!

**[Explore the BEHAVIOR-1K Website for more details!](https://behavior.stanford.edu/)**

## Key Features of BEHAVIOR-1K:

*   **1,000 Everyday Activities:** Test your agents on a massive range of tasks drawn from real-world human time-use data and preference studies.
*   **Realistic Simulation:** Built upon robust simulation environments for training and evaluation.
*   **Human-Centered Tasks:** Focus on practical skills like cooking, cleaning, and organization.
*   **Modular Installation:**  Install only the components you need for your specific research.
*   **Comprehensive Benchmark:** Provides a standardized platform for comparing and advancing embodied AI research.

## 🛠️ Installation

Get started with BEHAVIOR-1K quickly using our easy-to-follow installation instructions. The setup script handles all dependencies, allowing you to focus on your research.

### System Requirements

Ensure your system meets the following requirements:

*   **OS:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

The recommended approach for most users is a full installation within a new conda environment.

#### Linux

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --teleop --dataset
```

#### Windows

```powershell
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dataset
```

> **Note:** Run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation with these available components and additional options:

#### Available Components

| Component        | Flag           | Description                                       |
| ---------------- | -------------- | ------------------------------------------------- |
| OmniGibson       | `--omnigibson` | Core physics simulator and robotics environment     |
| BDDL             | `--bddl`       | Behavior Domain Definition Language for task specification |
| Teleoperation    | `--teleop`     | JoyLo interface for robot teleoperation             |

#### Additional Options

| Option                 | Flag                    | Description                                                              |
| ---------------------- | ----------------------- | ------------------------------------------------------------------------ |
| New Environment        | `--new-env`             | Create a new conda environment named `behavior`                          |
| Datasets               | `--dataset`             | Download BEHAVIOR datasets (requires `--omnigibson`)                    |
| Primitives             | `--primitives`          | Install OmniGibson with action primitives support                          |
| Development            | `--dev`                 | Install development dependencies                                        |
| CUDA Version           | `--cuda-version X.X`    | Specify CUDA version (default: 12.4)                                     |
| Conda TOS              | `--accept-conda-tos`    | Automatically accept Anaconda Terms of Service                           |
| NVIDIA EULA            | `--accept-nvidia-eula`  | Automatically accept NVIDIA Isaac Sim End User License Agreement         |
| Dataset License        | `--accept-dataset-tos`  | Automatically accept BEHAVIOR Data Bundle License Agreement            |

For automated/CI environments, you can bypass all prompts:
```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To see all available options:
```bash
./setup.sh --help
```

## 📄 Citation

If you use BEHAVIOR-1K in your research, please cite the following paper:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```

**[Back to the BEHAVIOR-1K Repository](https://github.com/StanfordVL/BEHAVIOR-1K)**