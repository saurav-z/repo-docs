# BEHAVIOR-1K: Unleash the Power of Embodied AI in Everyday Activities

[![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)](https://github.com/StanfordVL/BEHAVIOR-1K)

**BEHAVIOR-1K is a comprehensive simulation benchmark that challenges embodied AI agents to master 1,000 realistic household tasks.** This repository provides everything you need to train and evaluate agents on human-centered activities, mirroring real-world human behavior and preferences. Explore the full capabilities of BEHAVIOR-1K on the [main website](https://behavior.stanford.edu/).

**Key Features:**

*   **1,000 Everyday Activities:** Tackle a wide range of tasks, from cooking and cleaning to organizing and more.
*   **Human-Centered Tasks:** Activities are selected based on real-world human time-use surveys and preference studies.
*   **Realistic Simulation:** Built upon the powerful OmniGibson physics simulator for accurate and immersive environments.
*   **Modular Installation:** Install only the components you need with flexible setup options.
*   **Open-Source:** Leverage the power of open-source tools and resources.

## Installation

The `setup.sh` (Linux) and `setup.ps1` (Windows) scripts streamline the installation process. Choose a full or modular installation based on your needs.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

**Full Installation (Recommended):**

**Linux:**

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --teleop --dataset
```

**Windows (Run PowerShell as Administrator):**

```powershell
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dataset
```

> **Note:**  For Windows, set the execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Installation Options

The installation script offers modularity with these components:

| Component       | Flag          | Description                                             |
| :-------------- | :------------ | :------------------------------------------------------ |
| OmniGibson      | `--omnigibson` | Core physics simulator and robotics environment          |
| BDDL            | `--bddl`      | Behavior Domain Definition Language for task specification |
| Teleoperation   | `--teleop`    | JoyLo interface for robot teleoperation                 |

**Additional Configuration:**

| Option                      | Flag                        | Description                                                                     |
| :-------------------------- | :-------------------------- | :------------------------------------------------------------------------------ |
| New Environment             | `--new-env`                 | Create a new conda environment (requires conda)                                  |
| Datasets                    | `--dataset`                 | Download BEHAVIOR datasets (requires `--omnigibson`)                             |
| Primitives                  | `--primitives`              | Install OmniGibson with action primitives support                                |
| Development                 | `--dev`                     | Install development dependencies                                                |
| CUDA Version                | `--cuda-version X.X`        | Specify CUDA version (default: 12.4)                                              |
| No Conda Confirmation       | `--confirm-no-conda`        | Skip confirmation prompt when not in a conda environment                        |
| Accept Conda TOS           | `--accept-conda-tos`        | Automatically accept Anaconda Terms of Service (for automated installations)   |
| Accept NVIDIA EULA          | `--accept-nvidia-eula`      | Automatically accept NVIDIA Isaac Sim End User License Agreement (automated)  |
| Accept Dataset License    | `--accept-dataset-tos`      | Automatically accept BEHAVIOR Data Bundle License Agreement (automated)        |

**Installation without Conda:**  Omit the `--new-env` flag when using your existing Python environment.  You can also use `--confirm-no-conda` to skip prompts.

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