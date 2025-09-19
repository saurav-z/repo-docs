# XLeRobot: Affordable Embodied AI for Everyone

**Build your own dual-arm household robot for under $660, bringing advanced embodied AI to your fingertips!** [Explore the XLeRobot project on GitHub](https://github.com/Vector-Wangel/XLeRobot).

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-brown.svg)](README_CN.md)

[<img src="https://github.com/user-attachments/assets/f9c454ee-2c46-42b4-a5d7-88834a1c95ab" alt="XLeRobot" width="100%" />](https://xlerobot.readthedocs.io/en/latest/index.html)

[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter/X](https://img.shields.io/twitter/follow/VectorWang?style=social)](https://twitter.com/VectorWang2)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://xlerobot.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/badge/Discord-XLeRobot-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/bjZveEUh6F)

---

## Key Features

*   **Affordable**: Starting at just $660, significantly cheaper than comparable embodied AI platforms.
*   **Rapid Assembly**: Build your XLeRobot in under 4 hours, accelerating your entry into robotics.
*   **Dual-Arm Design**: Equipped with two arms for versatile manipulation tasks.
*   **Open Source**: Built upon the foundations of [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot), ensuring community support and extensibility.
*   **Simulation & Real-World Control**: Supports both simulated environments and real-world control via keyboard, Xbox controller, and Switch Joycon.
*   **Active Development**: Continuously updated with new features, demos, and tutorials, as evidenced by the release history below.
*   **Community Driven**: Join the XLeRobot community and collaborate with other developers.

---

## Recent Updates & News

*   **2025-09-09:** Developer Assembly kit (excluding battery and IKEA cart) ready for purchase in [China (Taobao) for **3699￥**](https://e.tb.cn/h.SZFbBgZABZ8zRPe?tk=ba514rTBRjQ) and [world-wide for **579\$**](https://shop.wowrobo.com/products/xlerobot-dual-arm-mobile-household-robot-kit?variant=47297659961561).
*   **2025-09-09:** Mentoring at the [Embodied AI Home Robot Hackathon](https://www.seeedstudio.com/embodied-ai-worldwide-hackathon-home-robot.html) (Oct 25–26, Bay Area) held by **SEEED x Nvidia x Huggingface**. [Register HERE](https://docs.google.com/forms/d/e/1FAIpQLSdYYDegdgIypxuGJNLcoc8kbdmU4jKgl49zg4X-107LAmBN4g/viewform).
*   **2025-08-30**: XLeRobot 0.3.0 Release with final outfit touch up and household chores showcase demos.
*   **2025-07-30**: Real-life control via keyboard/Xbox controller/Switch joycon.
*   **2025-07-08**: Simulation environment update.
*   **2025-07-01**: Documentation website launch.
*   **2025-06-13**: XLeRobot 0.2.0 hardware setup release.

---

## Estimated Costs

> [!NOTE]
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price (Buy all the parts yourself) | US      | EU      | CN      | IN       |
| :---------------------------------- | :------ | :------ | :------ | :------- |
| **Basic** (laptop, single RGB head cam)  | **~$660** | **~€680** | **~¥3999** | **~₹87000** |
| ↑ Stereo dual-eye RGB head cam           | +$30    | +€30    | +¥199    | +₹6550   |
| + RasberryPi                           | +$79    | +€79    | +¥399    | +₹7999   |
| ↑ RealSense RGBD head cam               | +$220   | +€230   | +¥1499   | +₹35726  |

---

## Getting Started

1.  **Buy Parts**: Refer to the detailed [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  **3D Print**: Follow the [3D printing](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html) instructions.
3.  **Assemble**: Put it all together with the [assembly guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html).
4.  **Software Setup**: Get your robot moving with the [software guide](https://xlerobot.readthedocs.io/en/latest/software/index.html).

---

## Contribute

We welcome contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Key Contributors:**

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   [Zhuoyi Lu](https://lzhuoyi.github.io/Zhuoyi_Lu.github.io/): RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Yiyang Huang: RL & VLA implementation (ongoing)
*   YCP: WebUI for remote control (ongoing)
*   [Lixing Zhang](lixingzhang.com): Hardware design improvements
*   Nicole Yue: Documentation website setup
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

## Disclaimer

> [!NOTE]
> You are responsible for any damages caused by building, buying, or using an XLeRobot based on this repository.