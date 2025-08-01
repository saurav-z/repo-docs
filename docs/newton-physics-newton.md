# Newton: GPU-Accelerated Physics Simulation for Robotics

**Accelerate your robotics research with Newton, a cutting-edge, GPU-accelerated physics simulation engine built for speed, differentiability, and extensibility.**  ([See the original repository](https://github.com/newton-physics/newton))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push Events - AWS GPU Tests](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml)

**⚠️ Prerelease Software ⚠️**

**Important:** Newton is currently in active alpha development. Expect frequent breaking changes and an evolving API.

## Key Features

*   **GPU-Accelerated Performance:** Leveraging NVIDIA Warp for high-speed physics computations.
*   **Differentiable Simulation:** Enables gradient-based optimization and control.
*   **Extensible Architecture:** Designed for user-defined customization and expansion.
*   **Integration with MuJoCo Warp:** Utilizes MuJoCo Warp as a primary backend for realistic physics.
*   **Targeted for Robotics:** Specifically designed for roboticists and simulation researchers.

## Overview

Newton extends and generalizes Warp's `warp.sim` module, providing a powerful platform for robotics simulation. By emphasizing GPU-based computation, differentiability, and extensibility, Newton facilitates rapid iteration and scalable robotics simulation.

## Supported By

Newton is developed and maintained by:

*   Disney Research
*   Google DeepMind
*   NVIDIA

## Getting Started

Refer to the [development guide](https://newton-physics.github.io/newton/development-guide.html) for instructions on setting up and using Newton.