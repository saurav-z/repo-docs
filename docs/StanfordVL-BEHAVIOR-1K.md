# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark for Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K** is a groundbreaking simulation benchmark, challenging embodied AI agents to master 1,000 realistic, everyday household activities, pushing the boundaries of AI in human-centered tasks.

**Explore the full details on our [main website](https://behavior.stanford.edu/)!**

## Key Features of BEHAVIOR-1K:

*   **Extensive Activity Coverage:** Tests agents on a vast range of human-centered tasks like cleaning, cooking, and organizing, mirroring real-world time-use surveys.
*   **Realistic Simulation:** Utilizes advanced simulation environments to provide a rich and immersive training and evaluation experience.
*   **Modular Installation:** Offers flexible installation options, allowing you to install only the components you need for your specific research or development.
*   **Human-Centered Design:** The tasks are carefully selected based on real-world human activity data, providing a relevant and challenging benchmark.
*   **Comprehensive Testing Ground:** Empowers researchers to train and evaluate agents across a wide variety of scenarios and tasks.

## Installation Guide

This guide provides instructions on how to install BEHAVIOR-1K. The installation script handles all dependencies and components, with modular installation options.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

For most users, we recommend a full installation using a new Conda environment:

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

#### Available Components

| Component         | Flag          | Description                                                         |
|-------------------|---------------|---------------------------------------------------------------------|
| **OmniGibson**    | `--omnigibson` | Core physics simulator and robotics environment                    |
| **BDDL**          | `--bddl`        | Behavior Domain Definition Language for task specification         |
| **Teleoperation** | `--teleop`      | JoyLo interface for robot teleoperation                            |

#### Additional Options

| Option               | Flag                         | Description                                                                     |
|----------------------|------------------------------|---------------------------------------------------------------------------------|
| **New Environment**    | `--new-env`                  | Create a new conda environment named `behavior`                                  |
| **Datasets**           | `--dataset`                  | Download BEHAVIOR datasets (requires `--omnigibson`)                            |
| **Primitives**       | `--primitives`               | Install OmniGibson with action primitives support                              |
| **Development**      | `--dev`                      | Install development dependencies                                                 |
| **CUDA Version**     | `--cuda-version X.X`        | Specify CUDA version (default: 12.4)                                            |

### Terms of Service & License Acceptance

The BEHAVIOR-1K installation may require you to accept various terms of service and license agreements. For interactive installations, you will be prompted to accept these terms. For non-interactive/automated installations, use the following flags:

| Option                 | Flag                          | Description                                                                 |
|------------------------|-------------------------------|-----------------------------------------------------------------------------|
| **Conda TOS**          | `--accept-conda-tos`          | Automatically accept Anaconda Terms of Service                                |
| **NVIDIA EULA**        | `--accept-nvidia-eula`        | Automatically accept NVIDIA Isaac Sim End User License Agreement              |
| **Dataset License**    | `--accept-dataset-tos`        | Automatically accept BEHAVIOR Data Bundle License Agreement                   |

For automated/CI environments, you can bypass all prompts:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To see all available options:
```bash
./setup.sh --help
```

## Citation

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```

---

**[Back to the Top](https://github.com/StanfordVL/BEHAVIOR-1K)**