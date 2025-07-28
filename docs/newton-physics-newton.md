# Newton: GPU-Accelerated Physics Simulation for Robotics

**Accelerate your robotics research with Newton, a cutting-edge, GPU-powered physics simulation engine.**  ([View the original repository](https://github.com/newton-physics/newton))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push Events - AWS GPU Tests](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml)

**⚠️ Important Note: This is prerelease software in active alpha development. The API is subject to change.**

## Key Features

Newton empowers robotics researchers with:

*   **GPU Acceleration:** Leverages the power of NVIDIA GPUs for significantly faster simulations.
*   **Differentiability:** Supports gradient calculations, enabling optimization and learning tasks.
*   **Extensibility:** Designed for user-defined customizations and integrations.
*   **Built on NVIDIA Warp:** Extends and enhances the capabilities of NVIDIA Warp's simulation module.
*   **MuJoCo Warp Integration:** Integrates MuJoCo Warp as a primary backend for realistic physics.
*   **Scalable Simulation:** Facilitates rapid iteration and handles large-scale robotics simulations efficiently.

##  What is Newton?

Newton is a physics simulation engine built on [NVIDIA Warp](https://github.com/NVIDIA/warp), specifically targeting roboticists and simulation researchers. It is developed by [Disney Research](https://www.disneyresearch.com/), [Google DeepMind](https://deepmind.google/), and [NVIDIA](https://www.nvidia.com/). Newton is focused on GPU-based computation, differentiability, and user-defined extensibility, facilitating rapid iteration and scalable robotics simulation.

## Development

For instructions on how to get started, please see the [development guide](https://newton-physics.github.io/newton/development-guide.html).