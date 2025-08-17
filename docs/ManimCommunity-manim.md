<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
    <br />
    <br />
    <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
    <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit"></a>
    <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter">
    <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"> </a>
    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
</p>

# Manim: Create Stunning Math Animations with Python

**Bring your mathematical concepts to life with Manim, a powerful animation engine for generating high-quality explanatory math videos.**  This community-driven version is a fork of the original Manim project created by Grant Sanderson (3Blue1Brown).

[**Visit the original repository for more information**](https://github.com/ManimCommunity/manim)

## Key Features

*   **Programmatic Animation:** Define animations using Python code for precise control.
*   **Versatile Scene Construction:** Build complex scenes with various objects and transformations.
*   **High-Quality Output:** Generate videos suitable for educational content and presentations.
*   **Community-Driven:** Benefit from an active community with improved features, documentation, and support.
*   **Interactive Development:**  Use the `%%manim` IPython magic within Jupyter notebooks for streamlined creation.
*   **Cross-Platform Support:** Available for local installation on various operating systems, and also available as a Docker image.

## Table of Contents

*   [Installation](#installation)
*   [Usage](#usage)
*   [Command Line Arguments](#command-line-arguments)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help and Community](#help-with-manim)
*   [Contributing](#contributing)
*   [How to Cite Manim](#how-to-cite-manim)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

## Installation

> [!CAUTION]
> These instructions are for the community version _only_.  For information about [3b1b/manim](https://github.com/3b1b/manim), please refer to their documentation.

To get started, you must install the necessary dependencies, but if you want to try it out before installing, you can do so [in our online Jupyter environment](https://try.manim.community/).

For local installation, please visit the [Documentation](https://docs.manim.community/en/stable/installation.html)
and follow the appropriate instructions for your operating system.

## Usage

Manim offers incredible versatility. Here's a basic example of how to create a simple scene:

```python
from manim import *

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        square.flip(RIGHT)
        square.rotate(-3 * TAU / 8)
        circle.set_fill(PINK, opacity=0.5)

        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
```

Save this code to a file (e.g., `example.py`) and run the following command in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will generate and preview a video transforming a square into a circle. Explore more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also integrates with JupyterLab and classic Jupyter notebooks through the `%%manim` IPython magic. Check the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) for guidance and try it out online [here](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

Manim's command-line interface provides several options.  Here's the general structure:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality for faster processing.
*   `-s`: Skip to the final frame.
*   `-n <number>`: Skip to the `n`th animation.
*   `-f`: Show the file in the file browser.

For a comprehensive list of command-line arguments, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

A Docker image (`manimcommunity/manim`) is available [on DockerHub](https://hub.docker.com/r/manimcommunity/manim). Instructions on its usage can be found in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help and Community

Get assistance with Manim or connect with the community via our [Discord Server](https://www.manim.community/discord/) and [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs or suggest features by opening an issue.

## Contributing

Contributions to Manim are encouraged, especially for tests and documentation. See the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines.

**Important:** Manim is currently undergoing a refactor.  New feature contributions may not be accepted during this period.  Join our [Discord server](https://www.manim.community/discord/) for the latest updates.

Most developers use `uv` for package management. Ensure that `uv` is installed. Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

If you use Manim in your work, please cite it to demonstrate its value.  Visit our [repository page](https://github.com/ManimCommunity/manim) and click the "cite this repository" button to generate a citation in your preferred format.

## Code of Conduct

Our code of conduct is available on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and Manim Community Developers (see LICENSE.community).