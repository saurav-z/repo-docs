# XLeRobot: Affordable AI for Your Home (and it's cheaper than an iPhone!)

[**Original Repository**](https://github.com/Vector-Wangel/XLeRobot)

XLeRobot is a groundbreaking, low-cost, dual-arm mobile robot designed to bring embodied AI to everyone, starting at just $660!

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

*   **Affordable:** Starting at $660, XLeRobot makes embodied AI accessible to a wider audience.
*   **Fast Assembly:**  Build your own XLeRobot in under 4 hours!
*   **Dual-Arm Design:** Enables complex manipulation tasks for household applications.
*   **Versatile Control:** Control XLeRobot with keyboard, Xbox controller, or Switch Joy-Con.
*   **Realistic Simulation:** Test and develop your robot in a simulated environment, including support for VR (Quest3).
*   **Comprehensive Documentation:** Get started quickly with organized tutorials, demos, and resources available on our documentation website.

## Cost Breakdown

> [!NOTE]
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price                                   | US      | EU      | CN       |
| :-------------------------------------- | :------ | :------ | :------- |
| **Basic** (use your laptop, single RGB head cam) | **~$660** | **~€680** | **~¥3999** |
| ↑ Stereo dual-eye RGB head cam         | +$30    | +€30    | +¥199    |
| + RasberryPi                          | +$79    | +€79    | +¥399    |
| ↑ RealSense RGBD head cam             | +$220   | +€230   | +¥1499   |

---

## What's New

*   **Real-life Control (July 30, 2025):** Control XLeRobot in real-time using a keyboard, Xbox controller, or Switch Joy-Con with zero latency.
    ![rea](https://github.com/user-attachments/assets/de8f50ad-a370-406c-97fb-fc01638d5624)

*   **Simulation Updates (July 08, 2025):** Experience updated URDFs, control scripts (Quest3 VR, keyboard, Xbox controller, and switch Joycon support), and RL environments. Get started in just 15 minutes!
    ![vr](https://github.com/user-attachments/assets/68b77bea-fdcf-4f42-9cf0-efcf1b188358)

*   **Documentation Website (July 01, 2025):** Discover organized tutorials, demos, and comprehensive resources on the newly launched documentation website.

*   **XLeRobot 0.2.0 Release (June 13, 2025):** The initial version fully capable for autonomous household tasks, starting from $660.

---

## Get Started

> [!NOTE]
> I'm a hardware rookie myself, so I want to make this tutorial friendly for all fellow beginners.

> [!NOTE]
> If you are totally new to programming, please spend at least a day to get yourself familiar with basic Python, Ubuntu and Github (with the help of Google and AI). At least you should know how to set up ubuntu system, git clone, pip install, use intepreters (VS Code, Cursor, Pycharm, etc.) and directly run commands in the terminals.

1.  💵 **Buy Parts:** Review the [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  🖨️ **3D Print:** Follow the [3D printing instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html).
3.  🔨 **Assemble:**  Put it all together using the [assembly guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html).
4.  💻 **Software:**  Get your robot moving with the [software setup](https://xlerobot.readthedocs.io/en/latest/software/index.html).

---

## Contributors

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   Zhuoyi Lu: RL sim2real deployment, teleop on real robot (Xbox, VR, Joycon)
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

**Acknowledgments:** This project builds on the shoulders of giants: [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).  Thank you to all the contributors to these amazing projects.

---

## Citation

If you want to cite this work:

```bibtex
@misc{wang2025xlerobot,
    author = {Wang, Gaotian}, {Lu, Zhuoyi}
    title = {XLeRobot: A Practical Low-cost Household Dual-Arm Mobile Robot Design for General Manipulation},
    howpublished = "\url{https://github.com/Vector-Wangel/XLeRobot}",
    year = {2025}
}
```

---

## Disclaimer

> [!NOTE]
> If you build, buy, or develop an XLeRobot based on this repository, you are fully responsible for any physical or mental damages it may cause.