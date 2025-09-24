# XLeRobot: Affordable and Accessible Embodied AI for Everyone

**Build your own dual-arm mobile robot for under $660 and start exploring the exciting world of embodied AI!** Learn more and contribute on the original repository: [XLeRobot GitHub](https://github.com/Vector-Wangel/XLeRobot).

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-brown.svg)](README_CN.md)
[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter/X](https://img.shields.io/twitter/follow/VectorWang?style=social)](https://twitter.com/VectorWang2)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://xlerobot.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/badge/Discord-XLeRobot-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/bjZveEUh6F)

<img src="https://github.com/user-attachments/assets/f9c454ee-2c46-42b4-a5d7-88834a1c95ab" alt="XLeRobot Overview" width="100%">

---

## Key Features of XLeRobot

*   **Low Cost:** Start building for as little as $660 USD, making embodied AI accessible to a wider audience.
*   **Dual-Arm Mobile Robot:** Explore advanced manipulation and interaction capabilities.
*   **Fast Assembly:** Assemble your robot in under 4 hours.
*   **Open Source:**  Built upon the foundations of [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), [Bambot](https://github.com/timqian/bambot) and fully open source, encouraging community contributions.
*   **Real-World Control:** Control XLeRobot with keyboard, Xbox controller, or Switch Joy-Cons.
*   **Simulation Support:** Get started quickly with comprehensive simulation environment.

---

## Recent Updates & News

*   **Hardware Assembly Video Tutorial** available on [YouTube](https://www.youtube.com/watch?v=upB1CEFeOlk) and [Bilibili](https://www.bilibili.com/video/BV1AGWFzUEJf/).
*   **Developer Assembly Kit Available:** Buy the hardware kit in [China (Taobao)](https://e.tb.cn/h.SZFbBgZABZ8zRPe?tk=ba514rTBRjQ) and [Worldwide](https://shop.wowrobo.com/products/xlerobot-dual-arm-mobile-household-robot-kit?variant=47297659961561).
*   **Embodied AI Home Robot Hackathon:** XLeRobot is a mentor at the [SEEED x Nvidia x Huggingface Embodied AI Hackathon](https://www.seeedstudio.com/embodied-ai-worldwide-hackathon-home-robot.html) (Oct 25–26, Bay Area). [Register HERE](https://docs.google.com/forms/d/e/1FAIpQLSdYYDegdgIypxuGJNLcoc8kbdmU4jKgl49zg4X-107LAmBN4g/viewform).
*   **XLeRobot 0.3.0 Release:**  Includes final outfit touch-ups and household chores showcase demos.
*   **Real-time Control:** Control XLeRobot in real life with keyboard, Xbox controller, or Switch Joy-Cons.
*   **Simulation Environment:** Updated simulation with VR support, controller support, and RL environments.
*   **Comprehensive Documentation:**  Check out the [XLeRobot documentation website](https://xlerobot.readthedocs.io/en/latest/index.html) for tutorials and resources.
<img src="https://github.com/user-attachments/assets/4132c23b-5c86-4bb9-94b4-a6b12059685b" width="100%">
---

## Cost Breakdown

> [!NOTE]
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price (Buy all the parts yourself) | US | EU | CN | IN |
| --- | --- | --- | --- | --- |
| **Basic** (use your laptop, single RGB head cam) | **~$660** | **~€680** | **~¥3999** | **~₹87000** |
| ↑ Stereo dual-eye RGB head cam | +$30 | +€30 | +¥199 | +₹6550 |
| + RasberryPi | +$79 | +€79 | +¥399 | +₹7999 |
| ↑ RealSense RGBD head cam | +$220 | +€230 | +¥1499 | +₹35726 |

---

## Getting Started Guide

1.  💵 **Buy Your Parts:**  Refer to the detailed [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  🖨️ **Print Your Parts:**  Follow the [3D Printing Guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html).
3.  🔨 **Assemble!**  Follow the [Assembly Instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html).
4.  💻 **Software Setup:** Get your robot moving with the [Software Guide](https://xlerobot.readthedocs.io/en/latest/software/index.html).

---

## Contribute

**We welcome contributions!**  See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Key Contributors:**

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   [Zhuoyi Lu](https://lzhuoyi.github.io/Zhuoyi_Lu.github.io/): RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Yiyang Huang: RL & VLA implementation (ongoing)
*   YCP: WebUI for remote control (ongoing)
*   [Lixing Zhang](lixingzhang.com): Hardware design improvements
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

---

## About the Creator

[Gaotian/Vector Wang](https://vector-wangel.github.io/), a CS graduate student at Rice University's [RobotPi Lab](https://robotpilab.github.io/), created XLeRobot to make embodied AI research and experimentation more accessible. His work focuses on robust object manipulation and closing the Sim2Real gap.

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
## ⚠️ Disclaimer ⚠️

> [!NOTE]
> You are responsible for any damages caused by building or using XLeRobot.