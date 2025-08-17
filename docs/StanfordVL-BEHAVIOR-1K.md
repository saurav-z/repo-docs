# BEHAVIOR-1K: Embodied AI for Everyday Household Tasks

**Tackle the complexities of the real world with BEHAVIOR-1K, a groundbreaking simulation benchmark featuring 1,000 human-centered household activities.** ([Original Repo](https://github.com/StanfordVL/BEHAVIOR-1K))

![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)

BEHAVIOR-1K provides a comprehensive platform for developing and evaluating embodied AI agents capable of performing a wide range of everyday tasks, such as cooking, cleaning, and organizing, mirroring real-world human activities.  Built with a focus on realism, the tasks are drawn from human time-use surveys and preference studies, making it ideal for training and testing the next generation of AI systems.

**Key Features:**

*   **Extensive Task Suite:** Access a benchmark of **1,000 diverse household activities**.
*   **Realistic Simulation:** Leverage an advanced physics engine for **realistic agent interactions**.
*   **Human-Centered Design:** Focus on tasks derived from **real-world human behavior**.
*   **Modular Installation:** Easily install only the components you need.
*   **Open Source:**  Freely use and extend the project to meet your research needs.

***For more detailed information, please visit our [main website](https://behavior.stanford.edu/)!***

## Installation

BEHAVIOR-1K offers a flexible installation process, enabling users to customize their setup with various components.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

The following commands provide a streamlined installation for most users, creating a new Conda environment:

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

Customize your installation with the following components and options:

#### Available Components

| Component       | Flag           | Description                                                 |
|-----------------|----------------|-------------------------------------------------------------|
| **OmniGibson**   | `--omnigibson`  | Core physics simulator and robotics environment            |
| **BDDL**         | `--bddl`        | Behavior Domain Definition Language for task specification |
| **Teleoperation** | `--teleop`      | JoyLo interface for robot teleoperation                      |

#### Additional Options

| Option                | Flag                      | Description                                                                |
|-----------------------|---------------------------|----------------------------------------------------------------------------|
| **New Environment**   | `--new-env`               | Create a new conda environment named `behavior`                             |
| **Datasets**          | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                         |
| **Primitives**        | `--primitives`            | Install OmniGibson with action primitives support                            |
| **Development**       | `--dev`                   | Install development dependencies                                            |
| **CUDA Version**      | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                        |
| **Conda TOS**         | `--accept-conda-tos`      | Automatically accept Anaconda Terms of Service                             |
| **NVIDIA EULA**       | `--accept-nvidia-eula`    | Automatically accept NVIDIA Isaac Sim End User License Agreement             |
| **Dataset License**   | `--accept-dataset-tos`    | Automatically accept BEHAVIOR Data Bundle License Agreement                  |

To bypass all interactive prompts in automated environments:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To see all available options, run:

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