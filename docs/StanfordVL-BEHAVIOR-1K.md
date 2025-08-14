# BEHAVIOR-1K: Embodied AI for Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**Unlock the potential of embodied AI with BEHAVIOR-1K, a comprehensive benchmark simulating 1,000 everyday household activities to train and evaluate intelligent agents.**  This repository provides the resources needed to build and test AI agents on tasks like cooking, cleaning, and organizing, mirroring real-world human time-use.

***For more details, visit our [main website](https://behavior.stanford.edu/)!***

## Key Features of BEHAVIOR-1K:

*   **1,000 Everyday Activities:** Train agents on a wide range of tasks derived from human time-use surveys.
*   **Realistic Simulation:**  Leverage the power of the OmniGibson physics simulator for accurate and engaging environments.
*   **Human-Centered Tasks:**  Focus on activities like cooking, cleaning, and organization, essential for practical embodied AI.
*   **Modular Installation:** Easily install only the components you need with the flexible setup script.
*   **Comprehensive Tooling:** Includes the Behavior Domain Definition Language (BDDL) for task specification and JoyLo interface for robot teleoperation.

## Installation Guide

Get started with BEHAVIOR-1K by following these steps.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ (Recommended)
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

We recommend a full installation within a new Conda environment for most users:

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

> **Note:**  Run PowerShell as Administrator and set the execution policy if necessary: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

The setup script offers various components and configurations:

#### Available Components

| Component        | Flag           | Description                                                |
|-----------------|----------------|------------------------------------------------------------|
| **OmniGibson**   | `--omnigibson` | Core physics simulator and robotics environment             |
| **BDDL**         | `--bddl`       | Behavior Domain Definition Language for task specification |
| **Teleoperation** | `--teleop`     | JoyLo interface for robot teleoperation                    |

#### Additional Options

| Option                | Flag                        | Description                                                               |
|-----------------------|-----------------------------|---------------------------------------------------------------------------|
| **New Environment**   | `--new-env`                 | Create a new Conda environment named `behavior`                           |
| **Datasets**          | `--dataset`                 | Download BEHAVIOR datasets (requires `--omnigibson`)                      |
| **Primitives**        | `--primitives`              | Install OmniGibson with action primitives support                         |
| **Development**       | `--dev`                     | Install development dependencies                                            |
| **CUDA Version**      | `--cuda-version X.X`        | Specify CUDA version (default: 12.4)                                      |
| **Conda TOS**         | `--accept-conda-tos`          | Automatically accept Anaconda Terms of Service                           |
| **NVIDIA EULA**       | `--accept-nvidia-eula`        | Automatically accept NVIDIA Isaac Sim End User License Agreement        |
| **Dataset License**   | `--accept-dataset-tos`        | Automatically accept BEHAVIOR Data Bundle License Agreement              |

For automated or CI environments, accept all terms automatically:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To view all available options, run:

```bash
./setup.sh --help
```

## Citation

If you use BEHAVIOR-1K, please cite our paper:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```

**Explore the original repository on GitHub: [https://github.com/StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)**