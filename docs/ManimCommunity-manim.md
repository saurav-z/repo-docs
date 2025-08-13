<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

## Manim: Create Stunning Math Animations with Python

**Manim** is a powerful Python library for generating high-quality animations perfect for explaining mathematical concepts, used by the popular [3Blue1Brown](https://www.3blue1brown.com/) channel.  Bring your mathematical ideas to life with dynamic visuals!  [Explore the original repository here](https://github.com/ManimCommunity/manim).

[![PyPI Latest Release](https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi)](https://pypi.org/project/manim/)
[![Docker Image](https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker)](https://hub.docker.com/r/manimcommunity/manim)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
[![MIT License](https://img.shields.io/badge/license-MIT-red.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit)](https://www.reddit.com/r/manim/)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community)](https://twitter.com/manim_community/)
[![Discord](https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord)](https://www.manim.community/discord/)
[![Black Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/manimce/badge/?version=latest)](https://docs.manim.community/)
[![Downloads](https://pepy.tech/badge/manim/month?)](https://pepy.tech/project/manim)
[![CI](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)](https://github.com/ManimCommunity/manim/workflows/CI)

<hr />

**Key Features:**

*   **Programmatic Animation:** Create precise and customizable animations using Python code.
*   **Versatile Scene Creation:** Build complex scenes with various mathematical objects and transformations.
*   **High-Quality Output:** Generate videos with sharp visuals and smooth animations.
*   **Community-Driven:** Benefit from an active community and ongoing development.
*   **Jupyter Integration:** Seamlessly integrate Manim into your Jupyter notebooks with the `%%manim` magic command.
*   **Docker Support:** Easily set up and run Manim using Docker containers.
*   **Online Exploration:**  Try Manim instantly with our interactive [Jupyter environment](https://try.manim.community/).

**Why Choose Manim Community Edition (ManimCE)?**

ManimCE is a community-maintained fork of the original Manim project by Grant Sanderson (3Blue1Brown). It offers:

*   **Active Development:**  Continuous improvements and new features.
*   **Enhanced Documentation:** Up-to-date and comprehensive documentation.
*   **Strong Community Support:**  A vibrant community providing help and resources.

## Table of Contents

*   [Installation](#installation)
*   [Usage](#usage)
*   [Command Line Arguments](#command-line-arguments)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help and Community](#help-with-manim)
*   [Contributing](#contributing)
*   [How to Cite](#how-to-cite-manim)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

## Installation

Manim requires dependencies and is available on several platforms.  For detailed installation instructions, consult the [official documentation](https://docs.manim.community/en/stable/installation.html).  You can also test it out without installation in our online [Jupyter environment](https://try.manim.community/).

## Usage

Here's a simple example of a scene:

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

Save this code as `example.py` and run it with:

```bash
manim -p -ql example.py SquareToCircle
```

This command will generate and preview a video demonstrating a square transforming into a circle.  Explore more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

## Command Line Arguments

Manim is typically used with the command-line interface, like this:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality for faster results.
*   `-s`: Show the final frame.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Show the output file in your file browser.

Detailed information on command line arguments is available in the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Manim provides a Docker image (`manimcommunity/manim`) on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Instructions are available in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help and Community

Get help and connect with the community via our [Discord Server](https://www.manim.community/discord/) and [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs and request features by opening an issue.

## Contributing

Contributions are welcomed!  Refer to the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines.  Join our [Discord server](https://www.manim.community/discord/) to discuss contributions and stay updated.

## How to Cite Manim

To acknowledge the value of Manim in your research and work, please cite the repository using the "cite this repository" button on the [repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Read the code of conduct and enforcement details on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and copyright by Manim Community Developers (see LICENSE.community).