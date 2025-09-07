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

# Manim: Create Stunning Math Videos with Python

**Bring your mathematical concepts to life with Manim, an open-source animation engine that lets you create dynamic and visually engaging math videos.**

<hr />

Manim is a powerful Python library for generating high-quality mathematical animations, perfect for educators, researchers, and anyone wanting to visualize complex ideas.  Learn more and contribute on the [Manim Community GitHub](https://github.com/ManimCommunity/manim).

**Key Features:**

*   **Precise Animations:** Programmatically create accurate and customizable animations.
*   **Python-Based:** Leverage the flexibility and power of Python for animation scripting.
*   **Community-Driven:** Benefit from a thriving community with active development and support.
*   **Versatile:**  Ideal for illustrating everything from basic algebra to advanced calculus and beyond.
*   **Cross-Platform:** Works on various operating systems, including Windows, macOS, and Linux.

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

To get started with Manim, you'll need to install its dependencies.  You can also try Manim directly [in our online Jupyter environment](https://try.manim.community/) without installing anything locally.

For detailed installation instructions, tailored to your operating system, please consult the [official documentation](https://docs.manim.community/en/stable/installation.html).

## Usage

Manim is designed to be intuitive. Here's a simple example to get you started:

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

Save this code in a file (e.g., `example.py`) and then, in your terminal, run:

```bash
manim -p -ql example.py SquareToCircle
```

This command will render a video transforming a square into a circle.  Explore more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also features a `%%manim` IPython magic for use in JupyterLab notebooks.  Find out more in the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and try it out online in our [Jupyter notebook](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

The general usage of Manim is:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the video automatically, while `-ql` renders it quickly at a lower quality.

Other useful flags:

*   `-s`:  Show the final frame only.
*   `-n <number>`:  Skip to the `n`th animation of a scene.
*   `-f`:  Show the output file in your file browser.

For a complete list of command-line options, please see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

The Manim community provides a Docker image (`manimcommunity/manim`) for convenient setup. Find it on [DockerHub](https://hub.docker.com/r/manimcommunity/manim). Installation and usage instructions can be found in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Need help? The Manim community is here to assist! Reach out on our [Discord Server](https://www.manim.community/discord/) or the [Reddit Community](https://www.reddit.com/r/manim/). Report bugs or suggest features by opening an issue on GitHub.

## Contributing

Contributions to Manim are highly encouraged!  The project is currently undergoing a major refactor, and new feature implementations are generally not being accepted at this time. However, if you're interested in contributing, especially to tests or documentation, consult the [documentation](https://docs.manim.community/en/stable/contributing.html) and join our [Discord server](https://www.manim.community/discord/) for discussion.

Most developers use `uv` for managing their Python environment. Instructions for installing manim with `uv` can be found in the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

Cite Manim in your research to acknowledge its value.  The best way to cite Manim is to use the "cite this repository" button on the [GitHub repository page](https://github.com/ManimCommunity/manim), which will generate a citation in your preferred format.

## Code of Conduct

Our Code of Conduct details how we treat each other. It can be read on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is dual-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and Manim Community Developers (see LICENSE.community).