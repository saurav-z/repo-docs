# BEHAVIOR-1K: Revolutionizing Embodied AI with 1,000 Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K is a comprehensive simulation benchmark designed to train and evaluate embodied AI agents on a wide range of human-centered tasks like cleaning, cooking, and organizing, mirroring real-world activities.** This repository provides everything you need to get started. Explore the [main website](https://behavior.stanford.edu/) for more details!

## Key Features

*   **Extensive Task Coverage:** Train agents on 1,000 diverse household activities.
*   **Realistic Simulation:** Utilize a high-fidelity simulation environment for accurate training and evaluation.
*   **Human-Centered Focus:** Tasks are derived from real-world human time-use surveys and preference studies.
*   **Modular Installation:**  Install only the components you need for your specific project.
*   **Teleoperation Support:** Includes a JoyLo interface for robot teleoperation.

## Installation Guide

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) and Windows 10+
*   **RAM:** 32GB+ (recommended)
*   **VRAM:** 8GB+ (minimum)
*   **GPU:** NVIDIA RTX 2080+ (or equivalent)

### Quick Start

For most users, we recommend a full installation. Choose either a new Conda environment or use your existing Python environment.

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

Customize your installation using the following flags:

#### Available Components

| Component      | Flag            | Description                                   |
| :------------- | :-------------- | :-------------------------------------------- |
| OmniGibson     | `--omnigibson`  | Core physics simulator and robotics environment |
| BDDL           | `--bddl`        | Behavior Domain Definition Language for task specification |
| Teleoperation  | `--teleop`      | JoyLo interface for robot teleoperation      |

#### Additional Options

| Option                       | Flag                   | Description                                                             |
| :--------------------------- | :--------------------- | :---------------------------------------------------------------------- |
| New Environment              | `--new-env`            | Create a new conda environment named `behavior` (requires conda)        |
| Datasets                     | `--dataset`            | Download BEHAVIOR datasets (requires `--omnigibson`)                   |
| Primitives                   | `--primitives`         | Install OmniGibson with action primitives support                       |
| Development                  | `--dev`                | Install development dependencies                                      |
| CUDA Version                 | `--cuda-version X.X`   | Specify CUDA version (default: 12.4)                                   |
| No Conda Confirmation        | `--confirm-no-conda`   | Skip confirmation prompt when not in a conda environment                |
| Conda TOS                    | `--accept-conda-tos`   | Automatically accept Anaconda Terms of Service                        |
| NVIDIA EULA                  | `--accept-nvidia-eula` | Automatically accept NVIDIA Isaac Sim End User License Agreement        |
| Dataset License              | `--accept-dataset-tos` | Automatically accept BEHAVIOR Data Bundle License Agreement             |

### Installation without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag:

#### Linux

```bash
./setup.sh --omnigibson --bddl --teleop --dataset
```

#### Windows

```powershell
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

To skip the confirmation prompt when not in a conda environment, add the `--confirm-no-conda` flag.

### Accepting Terms of Service & Licenses

For automated/CI environments, use these flags to bypass prompts:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

### Further Help

See all installation options with:

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

[View the original repository on GitHub](https://github.com/StanfordVL/BEHAVIOR-1K)