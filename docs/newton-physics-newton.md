[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push Events - AWS GPU Tests](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml)

# Newton: GPU-Accelerated Physics Simulation for Robotics

**Newton** is a cutting-edge, GPU-accelerated physics simulation engine designed for robotics and simulation research, providing unparalleled speed and flexibility.  **(See the original repository on GitHub: [https://github.com/newton-physics/newton](https://github.com/newton-physics/newton))**

**⚠️ Prerelease Software ⚠️**
*This project is in active alpha development.* This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

Developed with a focus on performance and extensibility, Newton leverages the power of NVIDIA Warp and integrates MuJoCo Warp to provide a robust and differentiable physics simulation environment.

**Key Features:**

*   **GPU-Accelerated:**  Achieve significant speedups with GPU-based computation, enabling faster simulations and rapid iteration.
*   **Differentiable:**  Supports differentiable physics, crucial for applications like reinforcement learning, optimization, and inverse design.
*   **Extensible:**  Designed with user-defined extensibility in mind, allowing you to easily customize and integrate new physics models.
*   **Robotics Focused:** Specifically targets the needs of roboticists and simulation researchers.
*   **MuJoCo Integration:** Includes MuJoCo Warp as a primary backend, benefiting from its established physics capabilities.
*   **Rapid Iteration:** Empowers researchers to quickly prototype, test, and refine their simulations.

**Developed By:**
Newton is maintained by [Disney Research](https://www.disneyresearch.com/), [Google DeepMind](https://deepmind.google/), and [NVIDIA](https://www.nvidia.com/).

## Getting Started

For detailed instructions on how to get started with Newton, refer to the [development guide](https://newton-physics.github.io/newton/development-guide.html).

## Examples

Explore the power of Newton with a variety of basic and advanced examples.  Run the examples directly from the command line.

### Basic Examples

| Example           | Command                                          |
| ----------------- | ------------------------------------------------ |
| Pendulum          | `python -m newton.examples basic_pendulum`        |
| URDF              | `python -m newton.examples basic_urdf`            |
| Viewer            | `python -m newton.examples basic_viewer`         |
| Shapes            | `python -m newton.examples basic_shapes`          |
| Joints            | `python -m newton.examples basic_joints`          |

### Cloth Examples

| Example           | Command                                        |
| ----------------- | ---------------------------------------------- |
| Cloth Bending     | `python -m newton.examples cloth_bending`      |
| Cloth Hanging     | `python -m newton.examples cloth_hanging`      |
| Cloth Style3D     | `python -m newton.examples cloth_style3d`      |

### MPM Examples

| Example           | Command                                         |
| ----------------- | ----------------------------------------------- |
| MPM Granular      | `python -m newton.examples mpm_granular`       |

## Example Options

The examples support the following common line arguments:

| Argument        | Description                                                                                         | Default                      |
| --------------- | --------------------------------------------------------------------------------------------------- | ---------------------------- |
| `--viewer`      | Viewer type: `gl` (OpenGL window), `usd` (USD file output), `rerun` (ReRun), or `null` (no viewer). | `gl`                         |
| `--device`      | Compute device to use, e.g., `cpu`, `cuda:0`, etc.                                                  | `None` (default Warp device) |
| `--num-frames`  | Number of frames to simulate (for USD output).                                                      | `100`                        |
| `--output-path` | Output path for USD files (required if `--viewer usd` is used).                                     | `None`                       |

Some examples may add additional arguments (see their respective source files for details).

## Example Usage

    # Basic usage
    python -m newton.examples basic_pendulum

    # With uv
    uv run python -m newton.examples basic_pendulum

    # With viewer options
    python -m newton.examples basic_viewer --viewer usd --output-path my_output.usd

    # With device selection
    python -m newton.examples basic_urdf --device cuda:0

    # Multiple arguments
    python -m newton.examples basic_viewer --viewer gl --num-frames 500 --device cpu
```
Key improvements and explanations:

*   **SEO-Optimized Heading:**  Uses "Newton: GPU-Accelerated Physics Simulation for Robotics" which is descriptive and includes relevant keywords.  This is the main selling point.
*   **One-Sentence Hook:**  Provides a concise and compelling introduction to the project, followed by a direct link to the original repository.
*   **Clearer Structure:** Uses headings, bullet points, and tables to improve readability and organization.
*   **Keyword Richness:**  Repeats key terms like "GPU-accelerated," "physics simulation," "robotics," and "differentiable" throughout the description.
*   **Concise Feature Descriptions:**  The key features are now presented as bullet points for easy scanning and comprehension.
*   **Actionable Examples:** The example section provides clear commands to run the examples.
*   **Added "Getting Started" section**: Added a section that points users directly to the development guide to help new users get started.
*   **Emphasis on Use Cases:** Highlights the benefits of differentiability and GPU acceleration for key applications.
*   **Warnings emphasized**: Important warnings about the prerelease stage are now clearly highlighted.
*   **Cleaned up formatting**: Improved readability and conciseness.