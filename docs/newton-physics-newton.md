# Newton: GPU-Accelerated Physics Simulation for Robotics & Research

**Unlock the power of rapid robotics simulation and research with Newton, a cutting-edge, GPU-accelerated physics engine.** (See the [original repo](https://github.com/newton-physics/newton) for more details.)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push Events - AWS GPU Tests](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml)

**⚠️ Prerelease Software ⚠️**

**Please note: This project is in active alpha development.** The API is subject to change.

## Key Features

Newton leverages the power of the GPU for efficient physics simulation, making it ideal for robotics and research applications:

*   **GPU Acceleration:** Built on NVIDIA Warp, enabling significant performance gains.
*   **Differentiability:** Supports differentiable simulations, facilitating gradient-based optimization and control.
*   **Extensibility:** Designed for user-defined extensions and customization to meet specific research needs.
*   **Integration with MuJoCo Warp:** Uses MuJoCo Warp as a primary backend for robust simulation capabilities.
*   **Rapid Iteration:** Empowers researchers to iterate quickly and scale their robotics simulations.

## Overview

Newton is a physics simulation engine designed specifically for robotics and simulation researchers. It extends and generalizes NVIDIA's Warp's existing `warp.sim` module. Newton also integrates MuJoCo Warp for its simulation backend. The project emphasizes GPU-based computation, differentiability, and user-defined extensibility, facilitating rapid iteration and scalable robotics simulation.

## Development

Get started with Newton by following the instructions in the [development guide](https://newton-physics.github.io/newton/development-guide.html).