<a href="https://xlerobot.readthedocs.io/en/latest/index.html">
  <img src="media/XLeRobot.png" alt="XLeRobot Logo" width="1200" />
</a>

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-brown.svg)](README_CN.md)

# XLeRobot: Affordable Open-Source Dual-Arm Mobile Robot

**XLeRobot makes embodied AI accessible to everyone with a dual-arm mobile robot that costs less than an iPhone and is assembled in under 4 hours!**  Explore the future of robotics with this open-source project built on the shoulders of giants.  Learn more and contribute on [GitHub](https://github.com/Vector-Wangel/XLeRobot).

[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter/X](https://img.shields.io/twitter/follow/VectorWang?style=social)](https://twitter.com/VectorWang2)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://xlerobot.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/badge/Discord-XLeRobot-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/bjZveEUh6F)

---

## Key Features

*   **Affordable:** Starting at just $660 (USD), making embodied AI more accessible.
*   **Fast Assembly:**  Build your robot in under 4 hours!
*   **Open Source:**  Leveraging [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot) for a collaborative approach.
*   **Comprehensive Documentation:**  Well-organized tutorials, demos, and resources available on the [XLeRobot Documentation website](https://xlerobot.readthedocs.io/en/latest/index.html).
*   **VR Control:** Explore advanced VR control with the XLeVR system, including Quest 3 whole-body control.
*   **Simulation Ready:** Official simulation environments with updated URDFs and control scripts.

<a href="https://xlerobot.readthedocs.io/en/latest/index.html">
  <img width="890" alt="XLeRobot in action" src="https://github.com/user-attachments/assets/c99fbd5f-af4a-48ba-a8fd-d667beec22c9" />
</a>

---

## 📰 What's New

*   **XLeVR (July 14, 2025):** Whole-body control system for VR with Quest 3, web-based, minimal dependencies, and modular design.
    ![vr](https://github.com/user-attachments/assets/68b77bea-fdcf-4f42-9cf0-efcf1b188358)

*   **Official Simulation (July 8, 2025):** Updated urdfs, control scripts and support for new hardware and cameras.

*   **XLeRobot Documentation Website (July 1, 2025):**  The official website is live, providing organized tutorials, demos, and resources.

*   **XLeRobot 0.2.0 (June 13, 2025):** Hardware setup and autonomous household tasks, starting at $660.

---

## 💵 Estimated Costs 💵

> [!NOTE]
> Costs exclude 3D printing, tools, shipping, and taxes.

| Configuration                                     | US         | EU          | CN          |
| :------------------------------------------------ | :--------- | :---------- | :---------- |
| **Basic** (no RasPi, use your own laptop)         | **~$660**  | **~€680**   | **~¥4000**  |
| **Standard** (RasPi, webcam head camera)         | **~$750**  | **~€770**   | **~¥4500**  |
| **Pro** (RasPi, RealSense RGBD head camera)      | **~$960**  | **~€980**   | **~¥6000**  |

---

## 🚀 Getting Started

> [!NOTE]
> This project is designed to be beginner-friendly.

> [!NOTE]
> If you're new to programming, consider spending a day familiarizing yourself with Python, Ubuntu, and Git.

1.  **Buy Your Parts:**  Refer to the [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  **3D Print:**  Follow the [3D printing guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html).
3.  **Assemble:**  Follow the [assembly instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html).
4.  **Software Setup:** Get your robot moving with the [software guide](https://xlerobot.readthedocs.io/en/latest/software/index.html).

---

## 🤝 Main Contributors

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   Zhuoyi Lu: RL sim2real deploy, VR control on real robot
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

This project is a collaborative effort, building upon the work of [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot). We welcome contributions!

[![Star History Chart](https://api.star-history.com/svg?repos=Vector-Wangel/XLeRobot&type=Timeline)](https://star-history.com/#Vector-Wangel/XLeRobot&Timeline)
---

## 📚 Citation

Cite this work using:

```bibtex
@misc{wang2025xlerobot,
    author = {Wang, Gaotian},
    title = {XLeRobot: A Practical Low-cost Household Dual-Arm Mobile Robot Design for General Manipulation},
    howpublished = "\url{https://github.com/Vector-Wangel/XLeRobot}",
    year = {2025}
}
```
---

## ⚠️ Disclaimer

> [!NOTE]
> You are solely responsible for any physical or mental damages resulting from building, buying, or developing an XLeRobot based on this repository.