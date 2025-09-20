# XLeRobot: Your Affordable Entry into Embodied AI Robotics

[XLeRobot](https://github.com/Vector-Wangel/XLeRobot) is an open-source, low-cost, dual-arm mobile robot designed to bring embodied AI to everyone, offering a hands-on learning experience for robotics enthusiasts and researchers.

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-brown.svg)](README_CN.md)
[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter/X](https://img.shields.io/twitter/follow/VectorWang?style=social)](https://twitter.com/VectorWang2)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://xlerobot.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/badge/Discord-XLeRobot-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/bjZveEUh6F)

<a href="https://xlerobot.readthedocs.io/en/latest/index.html">
  <img width="1725" height="1140" alt="front" src="https://github.com/user-attachments/assets/f9c454ee-2c46-42b4-a5d7-88834a1c95ab" />
</a>

## Key Features

*   **Affordable:** Starting at just \$660, XLeRobot offers an accessible entry point into robotics.
*   **Fast Assembly:** Build your own robot in under 4 hours.
*   **Dual-Arm Design:** Equipped with two arms for versatile manipulation tasks.
*   **Open-Source:** Built upon the giants [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), [Bambot](https://github.com/timqian/bambot).
*   **Real-World Control:** Control XLeRobot with keyboard, Xbox controller, or Switch Joy-Con for remote operation.
*   **Simulation Support:** Includes simulation environments to test and develop your robot's capabilities.
*   **Comprehensive Documentation:** Detailed documentation and tutorials are available to guide you through assembly, programming, and operation.

### [XLeRobot 0.3.0 Release](https://github.com/Vector-Wangel/XLeRobot)
*   Release with final outfit touch up and household chores showcase demos.

<img src="https://github.com/user-attachments/assets/17e31979-bd5e-4790-be70-566ea8bb181e" width="250"/>
<img src="https://github.com/user-attachments/assets/96ff4a3e-3402-47a2-bc6b-b45137ee3fdd" width="250"/>
<img src="https://github.com/user-attachments/assets/f6d52acc-bc8d-46f6-b3cd-8821f0306a7f" width="250"/>
<img src="https://github.com/user-attachments/assets/590863c1-3e6f-4a3c-b5e0-db893eeabc0c" width="250"/>
<img src="https://github.com/user-attachments/assets/4ddbc0ff-ca42-4ad0-94c6-4e0f4047fd01" width="250"/>
<img src="https://github.com/user-attachments/assets/7abc890e-9c9c-4983-8b25-122573028de5" width="250"/>
<img src="https://github.com/user-attachments/assets/e74a602b-0146-49c4-953d-3fa3b038a7f7" width="250"/>
<img src="https://github.com/user-attachments/assets/d8090b15-97f3-4abc-98c8-208ae79894d5" width="250"/>
<img src="https://github.com/user-attachments/assets/8b54adc3-d61b-42a0-8985-ea28f2e8f64c" width="250"/>

## News

*   **Developer Assembly Kit Now Available** The assembly kit (excluding battery and IKEA cart) is ready for purchase in [China (Taobao) for **3699￥**](https://e.tb.cn/h.SZFbBgZABZ8zRPe?tk=ba514rTBRjQ) and [world-wide for **579\$**](https://shop.wowrobo.com/products/xlerobot-dual-arm-mobile-household-robot-kit?variant=47297659961561).
*   **Hackathon Mentor:** XLeRobot is participating in the [Embodied AI Home Robot Hackathon](https://www.seeedstudio.com/embodied-ai-worldwide-hackathon-home-robot.html) (Oct 25–26, Bay Area) as a mentor.

## Getting Started

1.  **Parts:** Review the [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  **3D Printing:** Print the necessary components based on the [3D printing](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html) instructions.
3.  **Assembly:** Follow the [Assembly guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html).
4.  **Software:** Dive into the [Software](https://xlerobot.readthedocs.io/en/latest/software/index.html) section to get your robot moving!

## Cost Breakdown

> [!NOTE] 
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price (Buy all the parts yourself) | US | EU | CN | IN |
| --- | --- | --- | --- | --- |
| **Basic** (use your laptop, single RGB head cam) | **~$660** | **~€680** | **~¥3999** | **~₹87000** |
| ↑ Stereo dual-eye RGB head cam | +$30 | +€30 | +¥199 | +₹6550 |
| + RasberryPi | +$79 | +€79 | +¥399 | +₹7999 |
| ↑ RealSense RGBD head cam | +$220 | +€230 | +¥1499 | +₹35726 |

## Contribution

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Main Contributors**

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   [Zhuoyi Lu](https://lzhuoyi.github.io/Zhuoyi_Lu.github.io/): RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Yiyang Huang: RL & VLA implementation (ongoing)
*   YCP: WebUI for remote control (ongoing)
*   [Lixing Zhang](lixingzhang.com): Hardware design improvements
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

## Citation

```bibtex
@misc{wang2025xlerobot,
    author = {Wang, Gaotian and Lu, Zhuoyi},
    title = {XLeRobot: A Practical Low-cost Household Dual-Arm Mobile Robot Design for General Manipulation},
    howpublished = "\url{https://github.com/Vector-Wangel/XLeRobot}",
    year = {2025}
}
```
## Disclaimer
> [!NOTE]
> If you build, buy, or develop a XLeRobot based on this repo, you will be fully responsible for all the physical and mental damages it does to you or others.