<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
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
    <br />
    <br />
</p>

# Manim: Create Stunning Explanatory Math Videos with Python

Manim is a powerful Python library for creating mathematical animations, perfect for explaining complex concepts visually.  Explore the [Manim Community](https://github.com/ManimCommunity/manim) repository for more information.

**Key Features:**

*   **Programmatic Animation:** Generate precise animations using Python code, giving you complete control over every detail.
*   **Versatile Scene Creation:** Build custom scenes with shapes, text, and animations.
*   **Community Driven:** Benefit from an active community, with ongoing development and enhanced features compared to the original version.
*   **Clear Documentation:** Access comprehensive documentation for installation, usage, and customization.
*   **Integration:** Use Manim in Jupyter notebooks with the `%%manim` magic command.

**What is Manim?**

Manim (short for "Mathematical Animation Engine") is an animation engine that allows you to create explanatory math videos programmatically. This open-source project, maintained by the Manim Community, provides a flexible framework for crafting visually engaging animations, similar to those seen in 3Blue1Brown's educational videos.

**Why Choose ManimCE?**

ManimCE is the community-maintained version of Manim. We recommend this version for its:

*   Continued development
*   Improved features
*   Enhanced documentation
*   Active community-driven maintenance

If you're interested in the original version, you can find it at [3b1b/manim](https://github.com/3b1b/manim).

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

Before you begin, be sure to decide which version you wish to install.  These instructions are for the community version only. If you would like to study how Grant makes his videos, head over to his repository ([3b1b/manim](https://github.com/3b1b/manim)).

Manim requires several dependencies. To try it out first, use our [online Jupyter environment](https://try.manim.community/).

For local installation, consult the [Documentation](https://docs.manim.community/en/stable/installation.html) for OS-specific instructions.

## Usage

Manim is a versatile package. Below is an example Scene:

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

To see the output, save the code in a file (e.g., `example.py`) and run:

```bash
manim -p -ql example.py SquareToCircle
```

A video player will pop up with your animated scene. Further examples are found within this [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also supports `%%manim` magic for JupyterLab/Jupyter notebooks. See the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and try it [online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

The basic Manim command structure is:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

*   `-p`: Preview the video.
*   `-ql`: Render at a faster, lower quality.
*   `-s`: Skip to the final frame.
*   `-n <number>`: Go to the `n`th animation.
*   `-f`: Open the file browser after rendering.

For comprehensive options, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Find up-to-date documentation at [ReadTheDocs](https://docs.manim.community/).

## Docker

A Docker image (`manimcommunity/manim`) is available [on DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Installation and usage instructions are found in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help and Community

Get help with Manim and connect with other users through our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/). Report bugs or suggest features by opening an issue.

## Contributing

Contributions are welcome. See the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines.

Currently, new feature contributions are generally on hold during a major refactor. We highly recommend joining our [Discord server](https://www.manim.community/discord/) to discuss any potential contributions and to stay updated.

Most developers use `uv` for environment management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

Cite Manim in your work by visiting the [repository page](https://github.com/ManimCommunity/manim) and clicking "cite this repository" on the right sidebar.  This will generate a citation in your preferred format.

## Code of Conduct

Read our full code of conduct [on our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license (copyright by 3blue1brown LLC and Manim Community Developers). See `LICENSE` and `LICENSE.community`.