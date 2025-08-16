# BEHAVIOR-1K: The Ultimate Embodied AI Benchmark for Everyday Household Activities

![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)

**BEHAVIOR-1K** is a comprehensive simulation benchmark designed to challenge and advance embodied AI agents in performing realistic, everyday household tasks.  Explore the full details on our [main website](https://behavior.stanford.edu/)!

## Key Features

*   **1,000 Everyday Activities:** Test your agents on a wide range of tasks selected from real-world human time-use and preference studies, including cooking, cleaning, and organizing.
*   **Human-Centered Tasks:**  Focus on activities that humans commonly perform, pushing the boundaries of AI's understanding of human behavior and interaction with the world.
*   **Realistic Simulation:**  Built upon the robust OmniGibson physics simulator, providing a high-fidelity environment for training and evaluation.
*   **Modular Installation:** Easily install only the components you need using the flexible setup script.
*   **Teleoperation Support:** Utilize the JoyLo interface for robot teleoperation.

## Installation

Get started with BEHAVIOR-1K using our streamlined installation script.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

For most users, we recommend a full installation within a new conda environment:

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

> **Note:**  Run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

Customize your installation using the following components and options:

#### Available Components

| Component       | Flag           | Description                                         |
|-----------------|----------------|-----------------------------------------------------|
| **OmniGibson**  | `--omnigibson` | Core physics simulator and robotics environment    |
| **BDDL**        | `--bddl`       | Behavior Domain Definition Language for task specification |
| **Teleoperation** | `--teleop`       | JoyLo interface for robot teleoperation          |

#### Additional Options

| Option              | Flag                      | Description                                                                   |
|---------------------|---------------------------|-------------------------------------------------------------------------------|
| **New Environment** | `--new-env`               | Create a new conda environment named `behavior`                               |
| **Datasets**        | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                          |
| **Primitives**      | `--primitives`            | Install OmniGibson with action primitives support                              |
| **Development**     | `--dev`                   | Install development dependencies                                              |
| **CUDA Version**    | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                          |

### Terms of Service & License Acceptance

Automate your installation process by automatically accepting terms of service.

| Option                  | Flag                     | Description                                                                     |
|-------------------------|--------------------------|---------------------------------------------------------------------------------|
| **Conda TOS**           | `--accept-conda-tos`     | Automatically accept Anaconda Terms of Service                                    |
| **NVIDIA EULA**         | `--accept-nvidia-eula`   | Automatically accept NVIDIA Isaac Sim End User License Agreement                    |
| **Dataset License**     | `--accept-dataset-tos`   | Automatically accept BEHAVIOR Data Bundle License Agreement                         |

For automated or CI environments, bypass prompts with the following command:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To view all installation options:

```bash
./setup.sh --help
```

## Citation

If you use BEHAVIOR-1K in your research, please cite the following paper:

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```

**[Back to the original repository](https://github.com/StanfordVL/BEHAVIOR-1K)**