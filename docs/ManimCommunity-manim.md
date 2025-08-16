<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
    <br />
    <br />
    <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
    <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit" href=></a>
    <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter">
    <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"> </a>
    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
</p>

# Manim: Create Stunning Math Animations with Python

**Bring mathematical concepts to life with Manim, an open-source Python library for generating high-quality animations perfect for educational videos and visual explanations.**  Explore the [original Manim repository](https://github.com/ManimCommunity/manim) for more information.

## Key Features

*   **Programmatic Animation:** Generate animations by writing Python code, allowing for precise control and dynamic visuals.
*   **Versatile Scene Creation:** Design complex scenes with a wide range of geometric objects, transformations, and animation effects.
*   **Mathematical Visualization:**  Easily create animations of mathematical equations, functions, and geometric concepts.
*   **Community-Driven Development:** Benefit from an active community, extensive documentation, and continuous improvements.
*   **Cross-Platform Compatibility:** Use Manim on Windows, macOS, and Linux.
*   **Integration with Jupyter Notebooks:** Use Manim in Jupyter notebooks with the `%%manim` IPython magic.

## Table of Contents

*   [Installation](#installation)
*   [Usage](#usage)
*   [Command Line Arguments](#command-line-arguments)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help with Manim](#help-with-manim)
*   [Contributing](#contributing)
*   [How to Cite Manim](#how-to-cite-manim)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

## Installation

> [!CAUTION]
> These instructions are for the community version _only_. Trying to use these instructions to install [3b1b/manim](https://github.com/3b1b/manim) or instructions there to install this version will cause problems. Read [this](https://docs.manim.community/en/stable/faq/installation.html#why-are-there-different-versions-of-manim) and decide which version you wish to install, then only follow the instructions for your desired version.

Manim requires a few dependencies that must be installed prior to using it. If you
want to try it out first before installing it locally, you can do so
[in our online Jupyter environment](https://try.manim.community/).

For local installation, please visit the [Documentation](https://docs.manim.community/en/stable/installation.html)
and follow the appropriate instructions for your operating system.

## Usage

Create engaging mathematical animations by defining scenes in Python and running them. Here's a basic example:

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

This will generate a video showing a square transforming into a circle. Explore more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also integrates with JupyterLab (and classic Jupyter) notebooks via the `%%manim` IPython magic. See the [corresponding documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and [try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

The general usage of Manim is as follows:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag in the command above is for previewing, meaning the video file will automatically open when it is done rendering. The `-ql` flag is for a faster rendering at a lower quality.

Some other useful flags include:

*   `-s` to skip to the end and just show the final frame.
*   `-n <number>` to skip ahead to the `n`'th animation of a scene.
*   `-f` show the file in the file browser.

For a thorough list of command line arguments, visit the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

A Docker image (`manimcommunity/manim`) is maintained and available [on DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Find installation and usage instructions in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Get help with installation or usage on the [Discord Server](https://www.manim.community/discord/) or in the [Reddit Community](https://www.reddit.com/r/manim/). Report bugs and request features by opening an issue.

## Contributing

Contributions are welcome!  The project needs tests and documentation, especially.  See the [documentation](https://docs.manim.community/en/stable/contributing.html) for contribution guidelines.  Note that major refactoring is underway, and new feature contributions may be paused.  Join the [Discord server](https://www.manim.community/discord/) to discuss potential contributions.

Most developers on the project use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

To acknowledge the value of Manim in your work, cite the repository by using the "cite this repository" button on the right sidebar of the [repository page](https://github.com/ManimCommunity/manim). This generates citations in various formats.

## Code of Conduct

Review the full code of conduct and enforcement details on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is dual-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and the Manim Community Developers (see LICENSE.community).