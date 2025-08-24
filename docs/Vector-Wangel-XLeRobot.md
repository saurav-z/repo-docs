# XLeRobot: Affordable Embodied AI for Everyone

**XLeRobot is a groundbreaking project offering a low-cost, open-source, dual-arm mobile robot platform, bringing embodied AI within reach for hobbyists, researchers, and enthusiasts alike.**  [See the original project on GitHub](https://github.com/Vector-Wangel/XLeRobot).

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

*   **Cost-Effective:** Starts from approximately $660 USD, making embodied AI accessible.
*   **Fast Assembly:**  Assemble your XLeRobot in under 4 hours!
*   **Open-Source & Customizable:** Built upon open-source foundations [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), [Bambot](https://github.com/timqian/bambot), offering extensive customization options.
*   **Versatile Control:** Control your robot with a keyboard, Xbox controller, or Switch Joy-Con via Bluetooth.
*   **Simulation & Real-World Deployment:** Comprehensive support for simulation environments and real-world deployment, including support for Quest3 VR.
*   **Dual-Arm Capabilities:** Equipped with dual arms for complex manipulation tasks.
*   **Active Development:** Stay up-to-date with the latest features and improvements via the documentation, including detailed tutorials and demos.

![rea2](https://github.com/user-attachments/assets/e9b87163-9088-44a3-ac73-c23b6ba55f42)

## Cost Breakdown

> [!NOTE] 
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price | US | EU | CN |
| --- | --- | --- | --- |
| **Basic** (use your laptop, single RGB head cam) | **~$660** | **~€680** | **~¥3999** |
| ↑ Stereo dual-eye RGB head cam | +$30 | +€30 | +¥199 |
| + RasberryPi | +$79 | +€79 | +¥399 |
| ↑ RealSense RGBD head cam | +$220 | +€230 | +¥1499 |

---

## 📰 Recent Updates

*   **[Real-World Control](https://xlerobot.readthedocs.io/en/latest/software/index.html):** Control XLeRobot in real life via keyboard, Xbox controller, or Switch Joy-Con over Bluetooth with zero latency (July 30, 2025).
    ![rea](https://github.com/user-attachments/assets/de8f50ad-a370-406c-97fb-fc01638d5624)

*   **[Simulation Environment](https://xlerobot.readthedocs.io/en/latest/simulation/index.html):** Updated URDFs, control scripts (VR support via Quest3, keyboard, Xbox controller, Switch Joy-Con), and RL environments - get started in 15 minutes! (July 8, 2025).
    ![vr](https://github.com/user-attachments/assets/68b77bea-fdcf-4f42-9cf0-efcf1b188358)

*   **[Documentation Website](https://xlerobot.readthedocs.io/en/latest/index.html):**  New website with organized tutorials, demos, and resources (July 1, 2025).

*   **[XLeRobot 0.2.0](https://xlerobot.readthedocs.io):**  Hardware setup for the 1st version capable for autonomous household tasks, starting from $660 (June 13, 2025).

---

## 🚀 Getting Started

> [!NOTE] 
> Familiarity with basic Python, Ubuntu, and Git is recommended before getting started.

1.  💵 **Buy Parts:** Refer to the [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  🖨️ **3D Print:** Follow the [3D printing instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html).
3.  🔨 **Assemble:**  Assemble your robot using the [assembly guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html).
4.  💻 **Software:** Get your robot moving with the [software setup guide](https://xlerobot.readthedocs.io/en/latest/software/index.html).

---

## 🤝 Contribute

**Want to contribute to XLeRobot?** Review the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidance.

**Key Contributors:**

*   Zhuoyi Lu: RL sim2real deployment, real robot teleoperation (Xbox, VR, Joycon)
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

## Acknowledgements

Thanks to the open-source projects that made XLeRobot possible including [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).

## About the Creator

[Gaotian/Vector Wang](https://vector-wangel.github.io/)

I am a CS graduate student at Rice University, focusing on robust object manipulation. XLeRobot is built to instantiate my research theory, also to provide a low-cost platform for people who are interested in robotics and embodied AI to work with.

[![Star History Chart](https://api.star-history.com/svg?repos=Vector-Wangel/XLeRobot&type=Timeline)](https://star-history.com/#Vector-Wangel/XLeRobot&Timeline)

---

## 📚 Citation

If you use XLeRobot in your research, please cite it as:

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
> You are solely responsible for any physical or mental damages resulting from building, buying, or developing an XLeRobot based on this repository.