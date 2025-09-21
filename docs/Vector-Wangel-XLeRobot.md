# XLeRobot: Your Affordable Entry into Embodied AI Robotics

**XLeRobot is a DIY dual-arm mobile robot designed to bring embodied AI to everyone, offering advanced capabilities at a fraction of the cost of commercial robots.**  Explore the project on GitHub: [https://github.com/Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot)

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-brown.svg)](README_CN.md)

[<img width="1725" height="1140" alt="front" src="https://github.com/user-attachments/assets/f9c454ee-2c46-42b4-a5d7-88834a1c95ab" />](https://xlerobot.readthedocs.io/en/latest/index.html)

[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter/X](https://img.shields.io/twitter/follow/VectorWang?style=social)](https://twitter.com/VectorWang2)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://xlerobot.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/badge/Discord-XLeRobot-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/bjZveEUh6F)

---

## Key Features

*   **Cost-Effective:**  Starts at just $660, making advanced robotics accessible.
*   **Quick Assembly:** Build your robot in under 4 hours!
*   **Dual-Arm Functionality:** Enables complex manipulation tasks.
*   **Open Source:**  Leverages and builds upon the work of projects like [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), [Bambot](https://github.com/timqian/bambot), encouraging community contributions.
*   **Simulation & Real-World Control:**  Simulate in VR environments and control XLeRobot via keyboard, Xbox controller, or Switch Joy-Con.
*   **Comprehensive Documentation:**  Detailed guides and tutorials are available at [https://xlerobot.readthedocs.io/en/latest/index.html](https://xlerobot.readthedocs.io/en/latest/index.html).

---

## What's New

*   **[2025-09-09] Developer Assembly Kit Available:**  Purchase the assembly kit (excluding battery and IKEA cart) for [3699￥ (China)](https://e.tb.cn/h.SZFbBgZABZ8zRPe?tk=ba514rTBRjQ) and [579$ (Worldwide)](https://shop.wowrobo.com/products/xlerobot-dual-arm-mobile-household-robot-kit?variant=47297659961561) in collaboration with Wowrobo.
    *   This is an assembly kit for developers; please review documentation and the repo before purchase.
*   **[2025-09-09] Mentor at Embodied AI Home Robot Hackathon:** Mentoring at the SEEED x Nvidia x Huggingface hackathon (Bay Area, Oct 25-26). [Register HERE](https://docs.google.com/forms/d/e/1FAIpQLSdYYDegdgIypxuGJNLcoc8kbdmU4jKgl49zg4X-107LAmBN4g/viewform).
*   **[2025-08-30] XLeRobot 0.3.0 Release:** Featuring final outfit touch-ups and household chore demos.
*   **[2025-07-30] Real-Life Control:** Control XLeRobot with keyboard/Xbox controller/Switch joycon.
*   **[2025-07-08] Simulation:** Explore the simulation environment with updated URDFs, control scripts (VR, keyboard, gamepad), and more.  Get started in 15 minutes!
*   **[2025-07-01] Documentation Website:** Organized tutorials, demos, and resources are available at [https://xlerobot.readthedocs.io/en/latest/index.html](https://xlerobot.readthedocs.io/en/latest/index.html).
*   **[2025-06-13] XLeRobot 0.2.0 Release:** Hardware setup for autonomous household tasks, starting at $660.

---

## Cost Breakdown

> [!NOTE] 
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price (DIY) | US | EU | CN | IN |
| --- | --- | --- | --- | --- |
| **Basic** (laptop, single RGB cam) | **~$660** | **~€680** | **~¥3999** | **~₹87000** |
| ↑ Stereo dual-eye RGB head cam | +$30 | +€30 | +¥199 | +₹6550 |
| + RasberryPi | +$79 | +€79 | +¥399 | +₹7999 |
| ↑ RealSense RGBD head cam | +$220 | +€230 | +¥1499 | +₹35726 |

---

## Getting Started

1.  💵 **Buy Parts:**  Refer to the [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  🖨️ **3D Print:**  Follow the [3D printing guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html).
3.  🔨 **Assemble:**  Assemble your robot with the [assembly instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html).
4.  💻 **Software:**  Set up the software and get moving! [Software Guide](https://xlerobot.readthedocs.io/en/latest/software/index.html).

> [!NOTE]
>  Familiarize yourself with Python, Ubuntu, and Git.

---

## Contribute

**Contribute to XLeRobot!** See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

**Core Contributors:**

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   [Zhuoyi Lu](https://lzhuoyi.github.io/Zhuoyi_Lu.github.io/): RL sim2real deploy, teleop
*   Yiyang Huang: RL & VLA implementation (ongoing)
*   YCP: WebUI for remote control (ongoing)
*   [Lixing Zhang](lixingzhang.com): Hardware design
*   Nicole Yue: Documentation website
*   Yuesong Wang: Mujoco simulation

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

## About the Creator

[Gaotian/Vector Wang](https://vector-wangel.github.io/), a CS graduate student at Rice University's RobotPi Lab, built XLeRobot to explore robust object manipulation and make robotics more accessible.

[![Star History Chart](https://api.star-history.com/svg?repos=Vector-Wangel/XLeRobot&type=Timeline)](https://star-history.com/#Vector-Wangel/XLeRobot&Timeline)
---

## ⚠️ Disclaimer ⚠️

> [!NOTE]
> You are fully responsible for any physical or mental harm resulting from building, using, or developing XLeRobot based on this repository.