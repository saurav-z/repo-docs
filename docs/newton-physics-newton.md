# Newton: GPU-Accelerated Physics Simulation for Robotics & Research

**Accelerate your robotics simulations with Newton, a cutting-edge, GPU-powered physics engine.**  ([View the original repository](https://github.com/newton-physics/newton))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push Events - AWS GPU Tests](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml)

**⚠️ Prerelease Software ⚠️**

*Note: This project is in active alpha development. The API is unstable, and breaking changes are expected.*

## Key Features

Newton offers a powerful and flexible approach to physics simulation, optimized for robotics and research applications:

*   **GPU-Accelerated Performance:** Leverages the power of GPUs for significantly faster simulation speeds.
*   **Differentiable Physics:** Enables gradient-based optimization and control, crucial for advanced robotics tasks.
*   **Extensible Design:**  Built on [NVIDIA Warp](https://github.com/NVIDIA/warp) and integrates [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) for a flexible and customizable simulation environment.
*   **Robotics Focused:** Specifically designed to meet the needs of roboticists and simulation researchers.
*   **Rapid Iteration:** Facilitates quick experimentation and scaling for robotics simulation workloads.

##  Under the Hood

Newton extends and generalizes Warp's `warp.sim` module, providing a robust backend with a focus on GPU computation.

## Development

Get started with Newton! See the [development guide](https://newton-physics.github.io/newton/development-guide.html) for instructions on how to contribute and get up and running.

##  Maintained By

Newton is a collaborative project maintained by:

*   [Disney Research](https://www.disneyresearch.com/)
*   [Google DeepMind](https://deepmind.google/)
*   [NVIDIA](https://www.nvidia.com/)