[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push Events - AWS GPU Tests](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml)

# Newton: GPU-Accelerated Physics Simulation for Robotics and Research

**[Visit the original repository](https://github.com/newton-physics/newton) for the latest updates.**

Newton is a cutting-edge, GPU-accelerated physics simulation engine, designed for roboticists and simulation researchers seeking high performance and flexibility.  This project is in active alpha development, meaning the API is unstable.

## Key Features

*   **GPU-Accelerated:** Leverage the power of NVIDIA GPUs for blazing-fast simulations.
*   **Differentiable:** Enable gradient-based optimization and control.
*   **Extensible:** Build custom physics models and integrate with existing workflows.
*   **MuJoCo Integration:** Built upon MuJoCo Warp, providing a robust and widely used backend.
*   **Robotics-Focused:** Tailored for the specific needs of robotics research and development.
*   **Rapid Iteration:** Accelerate your research with fast simulation cycles.

## Core Technologies

*   **NVIDIA Warp:** Foundation for high-performance, GPU-based computation.
*   **MuJoCo Warp:** Provides a physics simulation backend.

## Development

Refer to the [installation guide](https://newton-physics.github.io/newton/guide/installation.html) to get started.

## Examples

Explore various example simulations to get started.

### Basic Examples

<table border="0">
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/basic/example_basic_pendulum.py">
        <img src="docs/images/examples/example_basic_pendulum.jpg" alt="Pendulum">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/basic/example_basic_urdf.py">
        <img src="docs/images/examples/example_basic_urdf.jpg" alt="URDF">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/basic/example_basic_viewer.py">
        <img src="docs/images/examples/example_basic_viewer.jpg" alt="Viewer">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>python -m newton.examples basic_pendulum</code>
    </td>
    <td align="center">
      <code>python -m newton.examples basic_urdf</code>
    </td>
    <td align="center">
      <code>python -m newton.examples basic_viewer</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/basic/example_basic_shapes.py">
        <img src="docs/images/examples/example_basic_shapes.jpg" alt="Shapes">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/basic/example_basic_joints.py">
        <img src="docs/images/examples/example_basic_joints.jpg" alt="Joints">
      </a>
    </td>
    <td align="center" width="33%">
      <!-- <a href="newton/examples/basic/example_basic_viewer.py">
        <img src="docs/images/examples/example_basic_viewer.jpg" alt="Viewer">
      </a> -->
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>python -m newton.examples basic_shapes</code>
    </td>
    <td align="center">
      <code>python -m newton.examples basic_joints</code>
    </td>
    <td align="center">
      <!-- <code>python -m newton.examples basic_viewer</code> -->
    </td>
  </tr>
</table>

### Cloth Examples

<table border="0">
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_cloth_bending.py">
        <img src="docs/images/examples/example_cloth_bending.jpg" alt="Cloth Bending">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_cloth_hanging.py">
        <img src="docs/images/examples/example_cloth_hanging.jpg" alt="Cloth Hanging">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_cloth_style3d.py">
        <img src="docs/images/examples/example_cloth_style3d.jpg" alt="Cloth Style3D">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>python -m newton.examples cloth_bending</code>
    </td>
    <td align="center">
      <code>python -m newton.examples cloth_hanging</code>
    </td>
    <td align="center">
      <code>python -m newton.examples cloth_style3d</code>
    </td>
  </tr>
</table>

### MPM Examples

<table border="0">
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/mpm/example_mpm_granular.py">
        <img src="docs/images/examples/example_mpm_granular.jpg" alt="MPM Granular">
      </a>
    </td>
    <td align="center" width="33%">
      <!-- Future MPM example -->
    </td>
    <td align="center" width="33%">
      <!-- Future MPM example -->
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>python -m newton.examples mpm_granular</code>
    </td>
    <td align="center">
      <!-- Future MPM example -->
    </td>
    <td align="center">
      <!-- Future MPM example -->
    </td>
  </tr>
</table>

### Selection Examples

<table border="0">
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/selection/example_selection_cartpole.py">
        <img src="docs/images/examples/example_selection_cartpole.jpg" alt="Selection Cartpole">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/selection/example_selection_materials.py">
        <img src="docs/images/examples/example_selection_materials.jpg" alt="Selection Materials">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/selection/example_selection_articulations.py">
        <img src="docs/images/examples/example_selection_articulations.jpg" alt="Selection Articulations">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>python -m newton.examples selection_cartpole</code>
    </td>
    <td align="center">
      <code>python -m newton.examples selection_materials</code>
    </td>
    <td align="center">
      <code>python -m newton.examples selection_articulations</code>
    </td>
  </tr>
</table>

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

*   **SEO Optimization:**  Added keywords like "GPU-accelerated," "physics simulation," "robotics," and "research" to improve search visibility.
*   **Clear Hook:** The opening sentence is now a compelling introduction to what Newton offers.
*   **Headings:**  Organized the content with clear headings for better readability and SEO.
*   **Key Features:**  Used bullet points to highlight the main benefits of the software, making it easier for users to understand what Newton does.
*   **Concise Language:** Simplified language to improve clarity.
*   **Emphasis on Target Audience:** Clearly stated the intended users (roboticists and researchers).
*   **Call to Action:**  The first sentence includes a call to visit the repo.
*   **More Descriptive Text:** Improved the descriptions in the key features.
*   **Code Highlighting:** Maintained the format of the code examples.
*   **Formatting:** Improved table formatting for readability.