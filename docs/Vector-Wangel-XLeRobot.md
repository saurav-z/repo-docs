# XLeRobot: Affordable Embodied AI for Everyone

**XLeRobot is a low-cost, dual-arm mobile robot designed to bring the power of embodied AI to your home, all for less than the price of a new smartphone!**  Learn more and contribute at the [original XLeRobot repo](https://github.com/Vector-Wangel/XLeRobot).

## Key Features

*   **Affordable:** Starting at just \$660, XLeRobot is significantly more affordable than many other embodied AI platforms.
*   **Fast Assembly:** Build your XLeRobot in under 4 hours.
*   **Dual-Arm Design:** Equipped with two arms for versatile manipulation and interaction.
*   **Open Source:** Built upon the shoulders of giants: [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).
*   **Real-World Control:** Control XLeRobot in real life with keyboard/Xbox controller/Switch joycon.
*   **Simulation Support:** Simulate XLeRobot's behavior with updated URDFs, control scripts, and RL environment.

## Cost Breakdown

XLeRobot offers different configurations to fit your budget:

| Configuration | Cost (USD) | Cost (EUR) | Cost (CNY) |
|---------------|------------|------------|------------|
| Basic (Laptop, Single RGB Cam)    | ~$660      | ~€680      | ~¥3999      |
| + Stereo RGB Cam     | +$30       | +€30       | +¥199       |
| + Raspberry Pi      | +$79       | +€79       | +¥399       |
| + RealSense RGBD Cam | +$220      | +€230      | +¥1499      |

*Note: Costs exclude 3D printing, tools, shipping, and taxes.*

## What's New

Stay updated on the latest XLeRobot developments:

*   **2025-08-30:** XLeRobot 0.3.0 Release with final outfit touch up and household chores showcase demos. Assembly kit ready for purchase soon, stay tuned!
*   **2025-07-30:** Control XLeRobot in real life with keyboard/Xbox controller/Switch joycon in the wild anywhere. All bluetooth, no wifi needed and zero latency.
*   **2025-07-08:** Simulation with updated URDFs, control scripts, and RL environment.
*   **2025-07-01:** Documentation website available for tutorials, demos, and resources.
*   **2025-06-13:** XLeRobot 0.2.0 hardware setup release.

## Get Started

Follow these steps to build your own XLeRobot:

1.  **Buy Parts:**  Consult the [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  **3D Print:**  Print the necessary components based on the [3D printing](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html) instructions.
3.  **Assemble:**  Follow the [assembly guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html).
4.  **Software:** Set up the software to get your robot moving, explained in the [software](https://xlerobot.readthedocs.io/en/latest/software/index.html) section.

## Contribute

We welcome contributions!  Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Main Contributors:**

*   Zhuoyi Lu: RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

## About the Creator

XLeRobot was created by [Gaotian/Vector Wang](https://vector-wangel.github.io/), a CS graduate student at Rice University focusing on robust object manipulation.

## Citation

If you use XLeRobot, please cite it as:

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