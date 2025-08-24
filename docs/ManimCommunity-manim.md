<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

## Manim: Create Stunning Math Animations with Code

**Bring your mathematical concepts to life** with Manim, a powerful and versatile animation engine perfect for creating engaging explanatory videos. This community-driven fork of the original tool by 3Blue1Brown empowers you to visualize complex ideas through precise, programmatic animations. Explore the original repo [here](https://github.com/ManimCommunity/manim).

<br />

[![PyPI Latest Release](https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi)](https://pypi.org/project/manim/)
[![Docker Image](https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker)](https://hub.docker.com/r/manimcommunity/manim)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
[![MIT License](https://img.shields.io/badge/license-MIT-red.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit)](https://www.reddit.com/r/manim/)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community)](https://twitter.com/manim_community/)
[![Discord](https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord)](https://www.manim.community/discord/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/manimce/badge/?version=latest)](https://docs.manim.community/)
[![Downloads](https://pepy.tech/badge/manim/month?)](https://pepy.tech/project/manim)
[![CI](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)](https://github.com/ManimCommunity/manim/workflows/CI)

<hr />

**Key Features:**

*   **Programmatic Animation:** Create animations using Python code for precise control.
*   **Mathematical Visualization:**  Easily visualize mathematical concepts, equations, and diagrams.
*   **Community-Driven:** Benefit from an active community, enhanced features, and improved documentation.
*   **Versatile Application:** Suitable for educational videos, presentations, and visual explanations.
*   **Jupyter Notebook Integration:** Seamlessly integrate Manim into your JupyterLab (and classic Jupyter) workflows using the `%%manim` IPython magic.
*   **Docker Support:** Utilize Docker for easy installation and consistent environments.

## Table of Contents:

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

> [!CAUTION]
> These instructions are for the community version _only_. Trying to use these instructions to install [3b1b/manim](https://github.com/3b1b/manim) or instructions there to install this version will cause problems. Read [this](https://docs.manim.community/en/stable/faq/installation.html#why-are-there-different-versions-of-manim) and decide which version you wish to install, then only follow the instructions for your desired version.

Manim requires a few dependencies that must be installed prior to using it. If you
want to try it out first before installing it locally, you can do so
[in our online Jupyter environment](https://try.manim.community/).

For local installation, please visit the [Documentation](https://docs.manim.community/en/stable/installation.html)
and follow the appropriate instructions for your operating system.

## Usage

Here's a basic example to get you started:

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

Save this code in a file named `example.py` and run it in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will render a square transforming into a circle. Find more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also ships with a `%%manim` IPython magic which allows to use it conveniently in JupyterLab (as well as classic Jupyter) notebooks. See the
[corresponding documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) for some guidance and
[try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the video, and `-ql` renders it at a lower quality. Some other useful flags include:

*   `-s`: Show the final frame.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Show the output file in the file browser.

For a full list, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Use the official Docker image (`manimcommunity/manim`) available [on DockerHub](https://hub.docker.com/r/manimcommunity/manim). Find installation and usage instructions in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help and Community

Get help and connect with other users on our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs or suggest features by opening an issue.

## Contributing

Contributions are welcome!  Refer to the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines, but please note the current major refactor may limit contributions. Discuss potential contributions on our [Discord server](https://www.manim.community/discord/) to stay updated.

Most developers on the project use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

Please cite Manim in your work to demonstrate its value. Use the "cite this repository" button on the [GitHub repository page](https://github.com/ManimCommunity/manim) to generate a citation in your preferred format.

## Code of Conduct

Review our full code of conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and Manim Community Developers (see LICENSE.community).