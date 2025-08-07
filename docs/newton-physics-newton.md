# Newton: A GPU-Accelerated Physics Simulation Engine for Robotics (Alpha)

**Accelerate your robotics research with Newton, a cutting-edge, GPU-powered physics simulation engine designed for speed, differentiability, and extensibility.**

**(This is a pre-release project in active development. Expect API changes.)**

[View the original repository on GitHub](https://github.com/newton-physics/newton)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push Events - AWS GPU Tests](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml)

## Key Features

*   **GPU Acceleration:** Leverages the power of NVIDIA GPUs for blazing-fast physics simulations.
*   **Differentiable Physics:** Supports differentiable computations, enabling gradient-based optimization and control.
*   **Extensible Architecture:** Designed for user-defined extensions and customizations to meet diverse research needs.
*   **MuJoCo Warp Integration:** Integrates MuJoCo Warp as a primary backend, providing a robust foundation.
*   **Targeted for Robotics:** Specifically tailored to the needs of roboticists and simulation researchers.
*   **Built on NVIDIA Warp:** Extends and generalizes the functionality of the NVIDIA Warp `warp.sim` module.

## About Newton

Newton is a physics simulation engine built upon NVIDIA Warp, focusing on GPU-based computation, differentiability, and user-defined extensibility. It's specifically tailored for robotics applications, allowing for rapid iteration and scalable simulations.  Newton utilizes [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) as a core backend.

Newton is developed and maintained by researchers at [Disney Research](https://www.disneyresearch.com/), [Google DeepMind](https://deepmind.google/), and [NVIDIA](https://www.nvidia.com/).

## Development

Get started with Newton by consulting the [development guide](https://newton-physics.github.io/newton/development-guide.html).