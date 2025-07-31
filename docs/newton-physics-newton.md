# Newton: GPU-Accelerated Physics Simulation for Robotics

**Accelerate your robotics research with Newton, a cutting-edge, GPU-powered physics simulation engine.**

[View the original repository on GitHub](https://github.com/newton-physics/newton)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push Events - AWS GPU Tests](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml)

**⚠️ Important: Pre-Release Software ⚠️**

*   **Alpha Development:** This project is in active alpha. The API is subject to change, and breaking updates are frequent.
*   **Use with Caution:** Expect instability and potential issues during this development phase.

## Key Features

Newton is a GPU-accelerated physics simulation engine designed for roboticists and simulation researchers. It is built on NVIDIA Warp and integrates MuJoCo Warp as a primary backend, focusing on:

*   **GPU Acceleration:** Leveraging the power of GPUs for fast and efficient physics simulations.
*   **Differentiability:** Enabling gradient-based optimization and control for advanced robotics applications.
*   **Extensibility:** Allowing users to define custom physics models and integrate new functionalities.
*   **Scalable Simulations:** Designed for rapid iteration and handling complex robotics scenarios.

## What is Newton?

Newton extends and generalizes Warp's existing `warp.sim` module.  It utilizes [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) as a core component, designed to provide a flexible and powerful framework for physics-based simulations, suitable for robotics research and development.

## Supported By

Newton is maintained by a collaboration of leading research organizations:

*   [Disney Research](https://www.disneyresearch.com/)
*   [Google DeepMind](https://deepmind.google/)
*   [NVIDIA](https://www.nvidia.com/)

## Getting Started

For information on how to start using and contributing to Newton, please refer to the [development guide](https://newton-physics.github.io/newton/development-guide.html).