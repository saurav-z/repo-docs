# BEHAVIOR-1K: The Ultimate Benchmark for Embodied AI

**BEHAVIOR-1K** offers a comprehensive, human-centered benchmark, challenging AI agents to master 1,000 everyday household activities within a realistic simulation environment. ([See the original repository](https://github.com/StanfordVL/BEHAVIOR-1K))

![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)

**Key Features:**

*   **1,000 Everyday Activities:** Test AI agents on a diverse range of tasks inspired by real-world human behavior.
*   **Realistic Simulation:**  Built on the robust OmniGibson physics simulator, providing a believable environment for agent training and evaluation.
*   **Human-Centered Design:** Activities selected from real-world time-use surveys and preference studies.
*   **Modular Installation:** Easily install only the components you need.
*   **Teleoperation Support:** Includes a JoyLo interface for robot teleoperation.
*   **Behavior Domain Definition Language (BDDL):** Supports task specification with BDDL.

## Installation

Get started with BEHAVIOR-1K quickly and easily with our comprehensive installation script. It supports both Linux and Windows operating systems.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

We recommend a full installation for most users:

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

## Installation Options

Customize your installation with these components and options:

### Available Components

| Component        | Flag            | Description                                       |
| ---------------- | --------------- | ------------------------------------------------- |
| OmniGibson       | `--omnigibson`  | Core physics simulator and robotics environment   |
| BDDL             | `--bddl`        | Behavior Domain Definition Language for task spec |
| Teleoperation    | `--teleop`      | JoyLo interface for robot teleoperation           |

### Additional Options

| Option                  | Flag                    | Description                                                                      |
| ----------------------- | ----------------------- | -------------------------------------------------------------------------------- |
| New Environment         | `--new-env`             | Create a new conda environment named `behavior` (requires conda)                 |
| Datasets                | `--dataset`             | Download BEHAVIOR datasets (requires `--omnigibson`)                            |
| Primitives              | `--primitives`          | Install OmniGibson with action primitives support                               |
| Development             | `--dev`                 | Install development dependencies                                                  |
| CUDA Version            | `--cuda-version X.X`    | Specify CUDA version (default: 12.4)                                             |
| No Conda Confirmation | `--confirm-no-conda`    | Skip confirmation prompt when not in a conda environment                       |
| Conda TOS               | `--accept-conda-tos`    | Automatically accept Anaconda Terms of Service                                 |
| NVIDIA EULA             | `--accept-nvidia-eula`  | Automatically accept NVIDIA Isaac Sim End User License Agreement                 |
| Dataset License         | `--accept-dataset-tos`  | Automatically accept BEHAVIOR Data Bundle License Agreement                        |

### Installation without Conda

If you want to use an existing Python environment, omit the `--new-env` flag:

```bash
# Linux
./setup.sh --omnigibson --bddl --teleop --dataset

# Windows
.\setup.ps1 -OmniGibson -BDDL -Teleop -Dataset
```

### Automating Installation

For automated or CI/CD environments, skip prompts by using these flags:

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

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```

***
[Visit our main website for more information](https://behavior.stanford.edu/).