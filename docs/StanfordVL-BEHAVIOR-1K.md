# BEHAVIOR-1K: Embodied AI for Everyday Tasks

**[BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) empowers researchers to train and evaluate embodied AI agents on a diverse set of 1,000 realistic household activities.**

![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)

BEHAVIOR-1K is a comprehensive simulation benchmark, designed to push the boundaries of embodied AI research.  Built around real-world human activities like cleaning, cooking, and organization, this platform offers a robust environment for developing intelligent agents capable of navigating and interacting with the world.

**Key Features:**

*   **1,000 Everyday Activities:** Test your agents across a wide range of tasks derived from human time-use surveys and preference studies.
*   **Realistic Simulation:** Built on the OmniGibson physics simulator, providing a robust and accurate environment.
*   **Human-Centered Tasks:**  Focus on activities directly relevant to human daily life, fostering the development of practical AI solutions.
*   **Modular Installation:** Easily install only the components you need, offering flexibility and control.
*   **Teleoperation Support:** Includes a JoyLo interface for robot teleoperation.
*   **BDDL Integration:** Utilize the Behavior Domain Definition Language (BDDL) for task specification.

***Learn more about BEHAVIOR-1K on our [main website](https://behavior.stanford.edu/)!***

## 🛠️ Installation

BEHAVIOR-1K provides a straightforward installation script that handles dependencies and components.

### System Requirements

*   **OS:** Linux (Ubuntu 20.04+), Windows 10+
*   **RAM:** 32GB+ recommended
*   **VRAM:** 8GB+
*   **GPU:** NVIDIA RTX 2080+

### Quick Start

For most users, we recommend the full installation within a new conda environment:

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

| Component        | Flag            | Description                                               |
| ---------------- | --------------- | --------------------------------------------------------- |
| **OmniGibson**   | `--omnigibson`  | Core physics simulator and robotics environment            |
| **BDDL**         | `--bddl`        | Behavior Domain Definition Language for task specification |
| **Teleoperation** | `--teleop`      | JoyLo interface for robot teleoperation                   |

#### Additional Options

| Option                | Flag                     | Description                                                                        |
| --------------------- | ------------------------ | ---------------------------------------------------------------------------------- |
| **New Environment**    | `--new-env`              | Create a new conda environment named `behavior`                                      |
| **Datasets**           | `--dataset`              | Download BEHAVIOR datasets (requires `--omnigibson`)                                |
| **Primitives**         | `--primitives`           | Install OmniGibson with action primitives support                                 |
| **Development**        | `--dev`                  | Install development dependencies                                                   |
| **CUDA Version**       | `--cuda-version X.X`     | Specify CUDA version (default: 12.4)                                               |

### Terms of Service & License Acceptance

BEHAVIOR-1K installation may require acceptance of various terms of service and license agreements. Use the following flags for non-interactive/automated installation:

| Option                  | Flag                       | Description                                                                      |
| ----------------------- | -------------------------- | -------------------------------------------------------------------------------- |
| **Conda TOS**           | `--accept-conda-tos`       | Automatically accept Anaconda Terms of Service                                  |
| **NVIDIA EULA**         | `--accept-nvidia-eula`     | Automatically accept NVIDIA Isaac Sim End User License Agreement                  |
| **Dataset License**     | `--accept-dataset-tos`     | Automatically accept BEHAVIOR Data Bundle License Agreement                        |

For automated/CI environments:

```bash
./setup.sh --new-env --omnigibson --bddl --teleop --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

To view all available options:

```bash
./setup.sh --help
```

## 📄 Citation

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```