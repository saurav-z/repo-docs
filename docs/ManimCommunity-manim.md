<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

<p align="center">
    <b>Create stunning math animations and explanatory videos with Manim!</b>
</p>

<p align="center">
    <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
    <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit"></a>
    <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter"></a>
    <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
    <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"></a>
    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
</p>

<hr />

Manim is a powerful Python-based animation engine designed to create visually engaging and explanatory videos, perfect for mathematics, science, and educational content.  This is the **Manim Community Edition (ManimCE)**, a community-driven fork of the original [3Blue1Brown/manim](https://github.com/3b1b/manim) project.

**Key Features:**

*   **Programmatic Animation:** Create animations with precise control using Python code.
*   **Versatile Scene Construction:** Build complex scenes with geometric objects, text, and mathematical formulas.
*   **Customizable Animations:**  Fine-tune animation effects like transformations, transitions, and effects.
*   **Integration with Jupyter Notebooks:**  Seamlessly create and preview animations within Jupyter environments.
*   **Active Community & Extensive Documentation:** Benefit from a supportive community and comprehensive documentation.
*   **Docker Support:** Easily set up and run Manim using Docker containers.

## Table of Contents

*   [Installation](#installation)
*   [Usage](#usage)
*   [Command Line Arguments](#command-line-arguments)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help & Community](#help-with-manim)
*   [Contributing](#contributing)
*   [How to Cite Manim](#how-to-cite-manim)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

## Installation

> [!CAUTION]
>  These instructions are for the Manim Community Edition (ManimCE) only. Do not use these instructions to install the original [3b1b/manim](https://github.com/3b1b/manim) or vice versa.

Manim requires some dependencies to be installed before use.  For a quick try, use our [online Jupyter environment](https://try.manim.community/).

For local installation, see the  [official Documentation](https://docs.manim.community/en/stable/installation.html).

## Usage

Manim is a versatile package for animations. Here is an example:

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

Save the code in a file like `example.py`.  Then, run in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

You'll see a video where a square transforms into a circle. Check the [GitHub repository](example_scenes) for more examples, and the [official gallery](https://docs.manim.community/en/stable/examples.html) for more advanced ones.

Manim also offers a `%%manim` IPython magic for easy use in JupyterLab and Jupyter notebooks. Read the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html), and try it out [online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

Manim is used as follows:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag opens the video automatically.  `-ql` gives faster, lower-quality rendering.

Other useful flags include:

*   `-s`:  Show the final frame.
*   `-n <number>`:  Skip to the `n`th animation.
*   `-f`: Show the file in the file browser.

For the full list of command-line arguments, consult the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available on [ReadTheDocs](https://docs.manim.community/).

## Docker

A Docker image (`manimcommunity/manim`) is available on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).
Find installation and usage instructions in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help & Community

Get help and connect with the community on our [Discord Server](https://www.manim.community/discord/) and [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs or make feature requests by opening an issue in the repository.

## Contributing

Contributions are welcome! Help is especially needed for tests and documentation. Check the [documentation](https://docs.manim.community/en/stable/contributing.html) for contribution guidelines.

*Important*: Manim is currently being refactored. Contributions that introduce new features may not be accepted during this time. Join the [Discord server](https://www.manim.community/discord/) to discuss potential contributions and stay updated.

Most developers use `uv` for package management. Make sure you have it installed.

Learn more about `uv` [here](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

To acknowledge Manim in your research, cite the repository from the [repository page](https://github.com/ManimCommunity/manim) by clicking the "cite this repository" button.

## Code of Conduct

Read our full code of conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

The software is dual-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).