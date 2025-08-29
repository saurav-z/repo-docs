# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark for Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K** is a groundbreaking simulation benchmark designed to evaluate and train embodied AI agents on a diverse range of human-centric tasks within a realistic household environment. This comprehensive resource, available on [GitHub](https://github.com/StanfordVL/BEHAVIOR-1K), provides everything needed to develop and test AI agents capable of performing complex activities like cleaning, cooking, and organizing, mirroring real-world human behaviors.

***For in-depth information, visit our [main website](https://behavior.stanford.edu/)!***

## Key Features

*   **Extensive Task Coverage:** Tackle 1,000 everyday household activities, sourced from human time-use surveys and preference studies.
*   **Realistic Simulation:** Leverage the powerful OmniGibson simulator for authentic physics and robotics interactions.
*   **Modular Installation:** Customize your setup with a modular installation script, allowing you to install only the components you need.
*   **Teleoperation Support:** Utilize a JoyLo interface for seamless robot teleoperation.
*   **BDDL Integration:** Benefit from the Behavior Domain Definition Language (BDDL) for robust task specification.

## Installation Guide

Follow these steps to get started with BEHAVIOR-1K:

### System Requirements

Ensure your system meets the following requirements:

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ (Recommended)
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

For most users, the full installation is recommended. You can either create a new Conda environment or utilize your existing Python environment.

**Linux:**

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --teleop --dataset
```

**Windows:**

```powershell
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dataset
```

> **Note:** Run PowerShell as Administrator and set the execution policy if necessary: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation using these components and options:

#### Available Components

| Component        | Flag             | Description                                  |
| :--------------- | :--------------- | :------------------------------------------- |
| OmniGibson       | `--omnigibson`   | Core physics simulator and robotics environment |
| BDDL             | `--bddl`         | Behavior Domain Definition Language        |
| Teleoperation    | `--teleop`       | JoyLo interface for robot teleoperation      |

#### Additional Options

| Option                     | Flag                      | Description                                                                     |
| :------------------------- | :------------------------ | :------------------------------------------------------------------------------ |
| New Environment            | `--new-env`               | Create a new Conda environment named `behavior` (requires Conda)                  |
| Datasets                   | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                             |
| Primitives                 | `--primitives`            | Install OmniGibson with action primitives support                               |
| Development                | `--dev`                   | Install development dependencies                                                |
| CUDA Version               | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                             |
| No Conda Confirmation      | `--confirm-no-conda`      | Skip confirmation prompt when not in a Conda environment                         |
| Conda TOS                  | `--accept-conda-tos`      | Automatically accept Anaconda Terms of Service                                    |
| NVIDIA EULA                | `--accept-nvidia-eula`    | Automatically accept NVIDIA Isaac Sim End User License Agreement                  |
| Dataset License            | `--accept-dataset-tos`    | Automatically accept BEHAVIOR Data Bundle License Agreement                       |

### Installation without Conda

To use your existing Python environment:

```bash
# Linux
./setup.sh --omnigibson --bddl --teleop --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

Skip the confirmation prompt:

```bash
./setup.sh --omnigibson --bddl --teleop --dataset --confirm-no-conda
```

### Automated Installation (CI/CD)

For non-interactive environments, automate installation by accepting all terms:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

### Help

See all installation options:

```bash
./setup.sh --help
```

## Citation

If you use BEHAVIOR-1K in your research, please cite our paper:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```