<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
    <br />
    <br />
</p>

## Manim: Create Stunning Math Animations with Code

**Manim** is a powerful and versatile Python library that allows you to generate beautiful, customizable animations for explanatory math videos, presentations, and educational content. Bring your mathematical concepts to life visually!  Explore the original repository on [GitHub](https://github.com/ManimCommunity/manim).

[![PyPI Latest Release](https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi)](https://pypi.org/project/manim/)
[![Docker Image](https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker)](https://hub.docker.com/r/manimcommunity/manim)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
[![MIT License](https://img.shields.io/badge/license-MIT-red.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit)](https://www.reddit.com/r/manim/)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community)](https://twitter.com/manim_community/)
[![Discord](https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord)](https://www.manim.community/discord/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/manimce/badge/?version=latest)](https://docs.manim.community/)
[![Downloads](https://pepy.tech/badge/manim/month?)](https://pepy.tech/project/manim)
[![CI](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)](https://github.com/ManimCommunity/manim/workflows/CI)

**Key Features:**

*   **Programmatic Animation:** Create animations using Python code, giving you precise control over every detail.
*   **Mathematical Visualization:** Easily visualize complex mathematical concepts, equations, and data.
*   **Customizable Scenes:** Build unique scenes with a wide range of shapes, objects, and transformations.
*   **Community-Driven:** Actively maintained and improved by a vibrant community of developers and users.
*   **Integration with Jupyter:** Seamlessly integrate Manim animations into Jupyter notebooks using the `%%manim` magic command.

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

Manim requires specific dependencies. You can explore the library in our online Jupyter environment ([try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)) or follow the installation instructions for your operating system found on the [Documentation](https://docs.manim.community/en/stable/installation.html).

## Usage

Here's a simple example to get you started:

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

Save this code as `example.py` and run it in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will create a video transforming a square into a circle.  Explore more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

## Command Line Arguments

The basic command structure is:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality for faster processing.

Other useful flags:

*   `-s`: Show the final frame.
*   `-n <number>`: Skip to the `n`th animation.
*   `-f`: Open the output file in the file browser.

For a complete list, refer to the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Use the official Docker image (`manimcommunity/manim`) from [DockerHub](https://hub.docker.com/r/manimcommunity/manim) for easy setup.  Find installation and usage instructions in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help and Community

Get support and connect with other users:

*   [Discord Server](https://www.manim.community/discord/)
*   [Reddit Community](https://www.reddit.com/r/manim/)

Report bugs or request features by opening an issue.

## Contributing

Contributions are welcome!  Consult the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines.
For development, make sure you have `uv` installed and available in your environment ([uv documentation](https://docs.astral.sh/uv/)) and follow the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html).  Note that the project is currently undergoing a refactor, so new feature contributions may be limited.

## How to Cite Manim

Please cite Manim in your work to demonstrate its value.  Use the "cite this repository" button on the [repository page](https://github.com/ManimCommunity/manim) to generate a citation in your preferred format.

## Code of Conduct

Read our full code of conduct at [our website](https://docs.manim.community/en/stable/conduct.html).

## License

The software is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).