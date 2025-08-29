# XLeRobot: Affordable Embodied AI for Everyone

**XLeRobot brings cutting-edge embodied AI to your home at an accessible price, starting at just $660 and assembled in under 4 hours!**  Explore the world of robotics with this open-source, dual-arm mobile robot platform.

[![](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![](https://img.shields.io/badge/lang-中文-brown.svg)](README_CN.md)

[<img width="1725" height="1140" alt="front" src="https://github.com/user-attachments/assets/f9c454ee-2c46-42b4-a5d7-88834a1c95ab" />](https://xlerobot.readthedocs.io/en/latest/index.html)

[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter/X](https://img.shields.io/twitter/follow/VectorWang?style=social)](https://twitter.com/VectorWang2)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://xlerobot.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/badge/Discord-XLeRobot-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/bjZveEUh6F)

---

## Key Features

*   **Cost-Effective:**  Build your own embodied AI robot for as low as $660, significantly cheaper than a new smartphone.
*   **Rapid Assembly:** Assemble your XLeRobot in under 4 hours, making it a quick and accessible project.
*   **Dual-Arm Mobile Robot:**  Equipped with dual arms for versatile manipulation tasks.
*   **Open-Source & Customizable:** Built upon the shoulders of giants like LeRobot, SO-100/SO-101, Lekiwi, and Bambot, allowing for easy modifications and contributions.
*   **Multiple Control Options:** Control XLeRobot with a keyboard, Xbox controller, or even a Nintendo Switch Joycon, with zero latency.
*   **Simulation Environment:**  Simulate your robot with updated URDFs and control scripts, including support for VR headsets and various input devices.
*   **Comprehensive Documentation:**  Explore detailed tutorials, demos, and resources on the [official documentation website](https://xlerobot.readthedocs.io/en/latest/index.html).

---

## What's New

*   **Real-World Control:**  Control your XLeRobot in real life using keyboard, Xbox controller, or Switch Joycon - all via Bluetooth, for zero-latency control.
*   **Enhanced Simulation:**  Updated URDFs, control scripts (Quest3 VR support, keyboard, Xbox controller, switch joycon), support for new hardware and cameras, and an RL environment are now available.

---

## Cost Breakdown

| Price                                       | US      | EU      | CN      |
| ------------------------------------------- | ------- | ------- | ------- |
| **Basic** (laptop, single RGB head cam)     | **~$660** | **~€680** | **~¥3999** |
| ↑ Stereo dual-eye RGB head cam              | +$30    | +€30    | +¥199   |
| + RasberryPi                                | +$79    | +€79    | +¥399   |
| ↑ RealSense RGBD head cam                  | +$220    | +€230    | +¥1499  |

---

## Getting Started

Follow these steps to build and operate your XLeRobot:

> [!NOTE]
> Familiarity with Python, Ubuntu, and Git is recommended.

1.  **Purchase Parts:**  Refer to the [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  **3D Print Components:**  Follow the [3D printing instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html).
3.  **Assemble Your Robot:**  Consult the [assembly guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html).
4.  **Install Software:**  Get your robot moving with the [software setup guide](https://xlerobot.readthedocs.io/en/latest/software/index.html).

---

## Contribute

**How to Contribute:**

*   Check out the [CONTRIBUTING.md](CONTRIBUTING.md) for guidance on getting involved.

**Main Contributors**

*   Zhuoyi Lu: RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

This project builds upon the work of [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).

---

## About the Creator

Developed by [Gaotian/Vector Wang](https://vector-wangel.github.io/), a CS graduate student at Rice University focusing on robust object manipulation. XLeRobot is a personal project aiming to make robotics and embodied AI more accessible.

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
> You are fully responsible for any physical or mental damages caused by building, buying, or developing an XLeRobot based on this repository.

---

**[Back to Original Repo](https://github.com/Vector-Wangel/XLeRobot)**