# XLeRobot: Affordable Open-Source Embodied AI Robot for Everyone

**XLeRobot is a revolutionary open-source project offering a dual-arm mobile robot platform for embodied AI research and exploration, costing less than an iPhone!**  Explore the possibilities of advanced robotics without breaking the bank.  View the original repository [here](https://github.com/Vector-Wangel/XLeRobot).

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-brown.svg)](README_CN.md)

<a href="https://xlerobot.readthedocs.io/en/latest/index.html">
  <img width="1725" height="1140" alt="front" src="https://github.com/user-attachments/assets/f9c454ee-2c46-42b4-a5d7-88834a1c95ab" />
</a>

[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter/X](https://img.shields.io/twitter/follow/VectorWang?style=social)](https://twitter.com/VectorWang2)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://xlerobot.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/badge/Discord-XLeRobot-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/bjZveEUh6F)
---

## Key Features

*   **Affordable:** Starting at approximately $660 USD, making embodied AI accessible.
*   **Rapid Assembly:** Build your robot in less than 4 hours!
*   **Open-Source:** Built upon open source projects [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), [Bambot](https://github.com/timqian/bambot), and fully customizable.
*   **Multiple Control Options:** Control your XLeRobot via keyboard, Xbox controller, or Switch Joy-Cons.
*   **Simulation Support:** Experiment with the robot in simulated environments using updated URDFs and control scripts.
*   **Extensive Documentation:** Comprehensive documentation with tutorials, demos, and resources available at [https://xlerobot.readthedocs.io/en/latest/index.html](https://xlerobot.readthedocs.io/en/latest/index.html).

![rea2](https://github.com/user-attachments/assets/e9b87163-9088-44a3-ac73-c23b6ba55f42)

## Cost Breakdown

> [!NOTE]
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price | US  | EU   | CN   |
| :---- | :-- | :--- | :--- |
| **Basic** (use your laptop, single RGB head cam) | **~$660** | **~€680** | **~¥3999** |
| ↑ Stereo dual-eye RGB head cam | +$30  | +€30   | +¥199  |
| + RasberryPi | +$79  | +€79   | +¥399  |
| ↑ RealSense RGBD head cam | +$220 | +€230  | +¥1499 |

---

## What's New

*   **Real-World Control (2025-07-30):** Control XLeRobot in real life with keyboard, Xbox controller, or Switch Joy-Cons, all via Bluetooth with zero latency.
    ![rea](https://github.com/user-attachments/assets/de8f50ad-a370-406c-97fb-fc01638d5624)
*   **Simulation Update (2025-07-08):** Explore updated URDFs, control scripts (supporting Quest3 VR, keyboard, Xbox controller, and Switch Joy-Cons), and RL environment.
    ![vr](https://github.com/user-attachments/assets/68b77bea-fdcf-4f42-9cf0-efcf1b188358)
*   **Documentation Website Launch (2025-07-01):**  A dedicated documentation website ([https://xlerobot.readthedocs.io/en/latest/index.html](https://xlerobot.readthedocs.io/en/latest/index.html)) providing tutorials, demos, and resources.
*   **XLeRobot 0.2.0 Release (2025-06-13):** The first version fully capable of autonomous household tasks, starting from $660.

---

## Getting Started

> [!NOTE]
> If you're new to programming, familiarize yourself with Python, Ubuntu, and Git before proceeding. Understanding system setup, cloning repositories, installing packages (pip), and running commands in terminals is essential.

1.  **Buy Parts:**  Refer to the [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  **3D Print:**  Follow the [3D printing instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html).
3.  **Assemble:**  [Assemble your robot!](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html)
4.  **Software Setup:**  [Get your robot moving!](https://xlerobot.readthedocs.io/en/latest/software/index.html)

---

## Contribute

**Contribute to XLeRobot!** See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Main Contributors**

*   Zhuoyi Lu: RL sim2real deployment, teleoperation on real robot (Xbox, VR, Joycon)
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

This project builds upon the work of [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).

## About the Developer

[Gaotian/Vector Wang](https://vector-wangel.github.io/), a CS graduate student at Rice University, focuses on robust object manipulation.

[![Star History Chart](https://api.star-history.com/svg?repos=Vector-Wangel/XLeRobot&type=Timeline)](https://star-history.com/#Vector-Wangel/XLeRobot&Timeline)
---

## Citation

```bibtex
@misc{wang2025xlerobot,
    author = {Wang, Gaotian and Lu, Zhuoyi},
    title = {XLeRobot: A Practical Low-cost Household Dual-Arm Mobile Robot Design for General Manipulation},
    howpublished = "\url{https://github.com/Vector-Wangel/XLeRobot}",
    year = {2025}
}
```
---

## Disclaimer

> [!NOTE]
> You are fully responsible for any physical or mental harm resulting from building or using XLeRobot.