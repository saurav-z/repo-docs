# Newton: GPU-Accelerated Physics Simulation for Robotics & Research

**Accelerate your robotics research with Newton, a cutting-edge, GPU-powered physics simulation engine designed for speed, differentiability, and extensibility.**  ( [Visit the original repository](https://github.com/newton-physics/newton) )

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push Events - AWS GPU Tests](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml)

**Important Note:** This project is in active alpha development. The API is subject to change.

## Key Features of Newton

*   **GPU-Accelerated Simulation:** Leverages the power of GPUs for significantly faster simulation speeds.
*   **Differentiable Physics:**  Enables gradient-based optimization and learning directly within the simulation.
*   **Extensible Design:** Built on NVIDIA Warp, allowing for user-defined customization and extension of the physics engine.
*   **MuJoCo Warp Integration:** Integrates with MuJoCo Warp as a primary backend for robust and established physics modeling.
*   **Targeted at Robotics & Simulation:** Specifically designed for roboticists and researchers working on advanced simulation applications.

## Overview

Newton extends and generalizes NVIDIA Warp's `warp.sim` module, using  [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) as a backend. The engine emphasizes GPU-based computation to facilitate rapid iteration and scalable robotics simulation. Newton is maintained by [Disney Research](https://www.disneyresearch.com/), [Google DeepMind](https://deepmind.google/), and [NVIDIA](https://www.nvidia.com/).

## Development

Get started with Newton by following the instructions in the [development guide](https://newton-physics.github.io/newton/development-guide.html).