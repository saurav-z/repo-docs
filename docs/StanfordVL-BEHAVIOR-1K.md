# BEHAVIOR-1K: Embodied AI for Everyday Life

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**Tackle the complexities of daily life with BEHAVIOR-1K, a comprehensive simulation benchmark designed to train and evaluate embodied AI agents on 1,000 realistic household activities.**  This benchmark, sourced from real-world human behavior data, provides a robust platform for advancing AI in tasks like cleaning, cooking, and organization.

**[Explore the BEHAVIOR-1K website for in-depth information!](https://behavior.stanford.edu/)**

## Key Features of BEHAVIOR-1K

*   **Extensive Task Coverage:** Train agents on a diverse set of 1,000 everyday household activities, representing a wide range of real-world scenarios.
*   **Human-Centered Design:**  Tasks are selected from human time-use surveys and preference studies, ensuring relevance and realism.
*   **Realistic Simulation:**  Utilizes the OmniGibson physics simulator to provide a high-fidelity environment for agent training and evaluation.
*   **Modular Installation:**  Install only the components you need, optimizing your setup for specific tasks.
*   **Flexible Deployment:**  Supports both Linux and Windows environments for broad accessibility.

## Installation Guide

Get started with BEHAVIOR-1K by following the steps below. The installation script simplifies the process, handling dependencies and environment setup.

### System Requirements

*   **Operating System:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start Installation (Recommended)

Install the latest stable release (v3.7.0) with all components:

#### Linux

```bash
# Clone the repository
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

#### Windows

```powershell
# Clone the repository
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Run the setup script
.\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset
```

**Note:** For the latest development features, clone the `main` branch instead.

### Installation Options

The setup script offers several options for customizing your installation:

#### Available Components

| Component         | Flag          | Description                                               |
|-------------------|---------------|-----------------------------------------------------------|
| OmniGibson        | `--omnigibson`| Core physics simulator and robotics environment           |
| BDDL              | `--bddl`      | Behavior Domain Definition Language for task specification  |
| JoyLo             | `--joylo`     | JoyLo interface for robot teleoperation                   |

#### Additional Options

| Option                      | Flag                      | Description                                                                 |
|-----------------------------|---------------------------|-----------------------------------------------------------------------------|
| New Environment             | `--new-env`               | Create a new conda environment named `behavior` (requires conda)            |
| Datasets                    | `--dataset`               | Download BEHAVIOR datasets (requires `--omnigibson`)                       |
| Primitives                  | `--primitives`            | Install OmniGibson with action primitives support                             |
| Evaluation                  | `--eval`                  | Install evaluation support for OmniGibson                                   |
| Development Dependencies    | `--dev`                   | Install development dependencies                                           |
| CUDA Version                | `--cuda-version X.X`      | Specify CUDA version (default: 12.4)                                        |
| No Conda Confirmation       | `--confirm-no-conda`      | Skip confirmation prompt when not in a conda environment                      |
| Accept Conda TOS           | `--accept-conda-tos`      | Automatically accept Anaconda Terms of Service                                |
| Accept NVIDIA EULA         | `--accept-nvidia-eula`    | Automatically accept NVIDIA Isaac Sim End User License Agreement            |
| Accept Dataset License     | `--accept-dataset-tos`    | Automatically accept BEHAVIOR Data Bundle License Agreement                 |

### Installation without Conda

If you prefer to use your existing Python environment, omit the `--new-env` flag:

#### Linux

```bash
./setup.sh --omnigibson --bddl --joylo --dataset
```

#### Windows

```powershell
.\setup.ps1 -OmniGibson -BDDL -JoyLo -Dataset
```

###  Automated Installation (CI/CD)

To avoid prompts in automated environments, use the following flags:

```bash
./setup.sh --new-env --omnigibson --bddl --joylo --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

### Get Help

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

**[Back to the BEHAVIOR-1K Repository](https://github.com/StanfordVL/BEHAVIOR-1K)**