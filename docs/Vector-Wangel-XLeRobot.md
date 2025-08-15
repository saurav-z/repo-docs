# XLeRobot: Build Your Own Affordable Embodied AI Robot

**Unlock the future of robotics with XLeRobot, a cutting-edge, open-source, dual-arm mobile robot you can build for as little as $660!**  ([Original Repository](https://github.com/Vector-Wangel/XLeRobot))

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

## Key Features & Benefits

*   **Cost-Effective:**  Build your own embodied AI robot for a fraction of the price of commercial options, starting at just $660.
*   **Easy Assembly:** Designed for beginners, with a total assembly time of under 4 hours.
*   **Open Source & Customizable:**  Leverage the power of open-source software and hardware, allowing for complete customization and experimentation.
*   **Advanced Capabilities:** Equipped with dual arms for general manipulation and household tasks.
*   **Multiple Control Options:** Control the robot in real life with keyboard, Xbox controller, or Switch Joy-Con.
*   **Simulation Support:** Get started quickly with comprehensive simulation environments, including VR support (Quest3).
*   **Comprehensive Documentation:**  Access detailed tutorials, demos, and resources to guide you through the build and development process.
*   **Community Driven:** Join a vibrant community to collaborate and contribute to the project.

## Cost Breakdown

> [!NOTE]
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price                     | US      | EU      | CN      |
| ------------------------- | ------- | ------- | ------- |
| **Basic** (w/ laptop)     | **~$660** | **~€680** | **~¥3999** |
| ↑ Stereo Dual-Eye RGB Cam | +$30    | +€30    | +¥199   |
| + RasberryPi              | +$79    | +€79    | +¥399   |
| ↑ RealSense RGBD Cam      | +$220   | +€230   | +¥1499  |

---

## What's New

*   **July 30, 2025:** Control XLeRobot in real life with keyboard/Xbox controller/Switch Joy-Con. No Wi-Fi needed, all Bluetooth, and zero latency.
    ![rea](https://github.com/user-attachments/assets/de8f50ad-a370-406c-97fb-fc01638d5624)
*   **July 8, 2025:** Simulation with updated urdfs, control scripts (support Quest3 VR, keyboard, Xbox controller, switch joycon), support for new hardware and cameras, RL environment. Get started in 15 min.
    ![vr](https://github.com/user-attachments/assets/68b77bea-fdcf-4f42-9cf0-efcf1b188358)
*   **July 1, 2025:** Launched the new documentation website with organized tutorials, demos and resources.
*   **June 13, 2025:** XLeRobot 0.2.0 hardware setup, the 1st version fully capable for autonomous household tasks, starts from 660$.

---

## Get Started

> [!NOTE]
> If you are totally new to programming, please spend at least a day to get yourself familiar with basic Python, Ubuntu and Github (with the help of Google and AI). At least you should know how to set up ubuntu system, git clone, pip install, use intepreters (VS Code, Cursor, Pycharm, etc.) and directly run commands in the terminals.

Follow these simple steps to build your own XLeRobot:

1.  💵 **Buy Your Parts:**  Refer to the [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  🖨️ **3D Print:**  Print the necessary components using the provided [3D printing](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html) instructions.
3.  🔨 **Assemble:**  Assemble the robot following the [assembly guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html).
4.  💻 **Software:**  Install the software and get your robot moving using the [software guide](https://xlerobot.readthedocs.io/en/latest/software/index.html).

---

## Contributors

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   Zhuoyi Lu: RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

This project is inspired by the work of [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).

---

## Citation

If you use this project, please cite it as follows:

```bibtex
@misc{wang2025xlerobot,
    author = {Wang, Gaotian}, {Lu, Zhuoyi}
    title = {XLeRobot: A Practical Low-cost Household Dual-Arm Mobile Robot Design for General Manipulation},
    howpublished = "\url{https://github.com/Vector-Wangel/XLeRobot}",
    year = {2025}
}
```
---

## ⚠️ Disclaimer ⚠️

> [!NOTE]
> If you build, buy, or develop a XLeRobot based on this repo, you will be fully responsible for all the physical and mental damages it does to you or others.