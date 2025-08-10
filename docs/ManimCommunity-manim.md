<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

<h1 align="center">Manim: Create Stunning Mathematical Animations with Code</h1>

**Manim is a powerful Python library for generating high-quality mathematical animations, making complex concepts visually accessible.**  Explore the original repository [here](https://github.com/ManimCommunity/manim).

<br/>

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

<hr />

Manim (Manim Community Edition) is an animation engine that empowers you to create dynamic and engaging math videos, inspired by the educational content of [3Blue1Brown](https://www.3blue1brown.com/).  This version is maintained by the community, offering active development, improved features, and a thriving community.

## Key Features

*   **Precise Animations:** Create animations with precise control using programmatic methods.
*   **Versatile:**  Suitable for a wide range of applications, from illustrating mathematical concepts to creating visually appealing presentations.
*   **Community-Driven:** Benefit from active development, enhanced documentation, and a supportive community.
*   **Cross-Platform:** Works on Windows, macOS, and Linux.
*   **Integration:** Supports Jupyter Notebooks with a custom magic command (`%%manim`).

## Table of Contents

-   [Installation](#installation)
-   [Usage](#usage)
-   [Command Line Arguments](#command-line-arguments)
-   [Documentation](#documentation)
-   [Docker](#docker)
-   [Help and Community](#help-with-manim)
-   [Contributing](#contributing)
-   [How to Cite Manim](#how-to-cite-manim)
-   [Code of Conduct](#code-of-conduct)
-   [License](#license)

## Installation

Before you start, if you would like to try it out first before installing it locally, you can do so
[in our online Jupyter environment](https://try.manim.community/).

Manim requires a few dependencies.  For local installation, refer to the detailed instructions in the [Documentation](https://docs.manim.community/en/stable/installation.html) for your operating system.

> [!CAUTION]
> These instructions are for the community version _only_. Trying to use these instructions to install [3b1b/manim](https://github.com/3b1b/manim) or instructions there to install this version will cause problems. Read [this](https://docs.manim.community/en/stable/faq/installation.html#why-are-there-different-versions-of-manim) and decide which version you wish to install, then only follow the instructions for your desired version.

## Usage

Here's a simple example demonstrating how to create a scene:

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

Save the code as `example.py` and run it in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will render the scene, transforming a square into a circle.  Explore more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim integrates with JupyterLab/Jupyter notebooks using the `%%manim` magic command (see the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html)) and [try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

The general structure of a Manim command is:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the video, and `-ql` renders at a lower quality for faster previews.

Useful flags:

*   `-s`:  Shows the final frame.
*   `-n <number>`: Skips to the `n`th animation.
*   `-f`:  Opens the file in the file browser.

For a comprehensive list, consult the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

A Docker image (`manimcommunity/manim`) is maintained on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Find installation and usage instructions in our [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help and Community

Get help, ask questions, and connect with the community on our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs and suggest features by opening an issue.

## Contributing

Contributions are welcome!  We especially need help with tests and documentation. For contribution guidelines, please see the [documentation](https://docs.manim.community/en/stable/contributing.html).

However, please note that Manim is currently undergoing a major refactor. In general,
contributions implementing new features will not be accepted in this period.
The contribution guide may become outdated quickly; we highly recommend joining our
[Discord server](https://www.manim.community/discord/) to discuss any potential
contributions and keep up to date with the latest developments.

Most developers on the project use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

To acknowledge Manim in your work, cite the repository using the "cite this repository" button on the right sidebar of the [GitHub repository](https://github.com/ManimCommunity/manim).  This generates citations in various formats.

## Code of Conduct

Our Code of Conduct is available on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

This software is dual-licensed under the MIT license (copyright by 3blue1brown LLC and Manim Community Developers).  See `LICENSE` and `LICENSE.community`.