<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

# Manim: Create Stunning Explanatory Math Videos with Code

**Bring your mathematical concepts to life with Manim, a powerful and versatile animation engine used to generate high-quality videos.** This community-driven version of Manim allows you to create intricate animations programmatically, similar to those seen in the popular 3Blue1Brown educational math videos. Check out the original repo [here](https://github.com/ManimCommunity/manim).

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

**Key Features:**

*   **Programmatic Animation:** Create animations using Python code, offering precise control and flexibility.
*   **Mathematical Visualization:** Visualize complex mathematical concepts with ease.
*   **Community-Driven:** Benefit from an active community, continuous development, and enhanced documentation.
*   **Versatile Application:** Use Manim for educational videos, presentations, and more.
*   **Integration:** Seamlessly integrate Manim with Jupyter notebooks for interactive development.

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

Manim requires specific dependencies for proper functionality. We recommend you try it out in the [online Jupyter environment](https://try.manim.community/) before local installation.

Detailed installation instructions for various operating systems are available in the [official Documentation](https://docs.manim.community/en/stable/installation.html).
> [!CAUTION]
> Ensure you are following the instructions for the community version (ManimCE). Avoid using instructions from the [3b1b/manim](https://github.com/3b1b/manim) repository.

## Usage

Here's an example of a `Scene` demonstrating the transformation of a square into a circle:

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

Save the code as `example.py` and then run the following command in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This command will render and automatically preview the animation. For further examples, explore the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).  Manim can also be used within Jupyter notebooks with the `%%manim` IPython magic; documentation is available [here](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) with a live example [here](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

The basic structure for using Manim via the command line is as follows:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag opens the rendered video, and `-ql` provides a quick, lower-quality render. Additional useful flags include:

*   `-s`: Show the final frame.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Open the file in the file browser.

For a complete list of command line arguments, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available on [ReadTheDocs](https://docs.manim.community/).

## Docker

A Docker image (`manimcommunity/manim`) is maintained and available on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Installation and usage instructions can be found in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help & Community

Get help and connect with the community through the [Discord Server](https://www.manim.community/discord/) and [Reddit Community](https://www.reddit.com/r/manim/). For bug reports or feature requests, please submit an issue.

## Contributing

Contributions to Manim are welcome.  We particularly need help with tests and documentation. For contribution guidelines, see the [documentation](https://docs.manim.community/en/stable/contributing.html).

> Note: The project is undergoing a major refactor, and feature contributions may be limited. Join the [Discord server](https://www.manim.community/discord/) to stay updated on the latest developments.

Many developers use `uv` for package management. Install uv using its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

To ensure the value of Manim is recognized, please cite it in your work.  You can generate a citation in your preferred format by clicking the "cite this repository" button on the right sidebar of the [GitHub repository](https://github.com/ManimCommunity/manim).

## Code of Conduct

Read our full code of conduct, and how we enforce it, on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

The software is double-licensed under the MIT license (copyright by 3blue1brown LLC and Manim Community Developers). See `LICENSE` and `LICENSE.community` for details.