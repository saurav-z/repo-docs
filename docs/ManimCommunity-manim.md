<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
</p>
<h1 align="center">Manim: Create Stunning Math Animations</h1>

Manim is a powerful Python library for generating high-quality mathematical animations, perfect for educational videos and visual explanations. Developed by the Manim Community, this project provides a comprehensive toolset to bring your mathematical concepts to life.  Find the original source code and learn more at the [Manim repository](https://github.com/ManimCommunity/manim).

<p align="center">
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

<hr />

**Key Features:**

*   **Precise Animations:** Create animations programmatically with fine-grained control over every detail.
*   **Versatile:** Perfect for explaining mathematical concepts, visualizing data, and producing engaging educational content.
*   **Community-Driven:** Benefit from an active community, regular updates, and extensive documentation.
*   **Python-Based:**  Leverage the power of Python for scripting and customizing your animations.
*   **Cross-Platform:** Works on various operating systems, including Windows, macOS, and Linux.
*   **Docker Support:** Easily set up and run Manim using Docker containers.
*   **Jupyter Notebook Integration:** Utilize Manim directly within Jupyter notebooks for interactive development.

## Table of Contents:

-   [Installation](#installation)
-   [Usage](#usage)
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

Before you can use Manim, ensure you've installed the necessary dependencies.  If you'd like to experiment before installing, try our [online Jupyter environment](https://try.manim.community/).

For local installation, consult the [official documentation](https://docs.manim.community/en/stable/installation.html) and follow the instructions specific to your operating system.

## Usage

Manim is incredibly flexible. Here's a simple example `Scene`:

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

Save this code to a file (e.g., `example.py`).  Then, in your terminal, run:

```sh
manim -p -ql example.py SquareToCircle
```

This command will render and preview the animation of a square transforming into a circle.  You can find more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also supports the `%%manim` IPython magic for convenient use in JupyterLab (and classic Jupyter) notebooks. Refer to the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and the [online example](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

Here's the general structure for using Manim:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality (faster).
*   `-s`: Show the final frame only.
*   `-n <number>`:  Skip to the nth animation.
*   `-f`: Open the output file in your file browser.

For a complete list of arguments, see the [configuration documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

The Manim Community provides a Docker image (`manimcommunity/manim`) on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Installation and usage instructions are found in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help and Community

Get help with installation and usage through our [Discord Server](https://www.manim.community/discord/) or the [Reddit Community](https://www.reddit.com/r/manim). Submit bug reports and feature requests by opening an issue.

## Contributing

Contributions are highly encouraged!  We particularly need help with testing and documentation.  See the [contributing guidelines](https://docs.manim.community/en/stable/contributing.html).

Please note that Manim is currently undergoing a major refactor.  New feature contributions are generally paused during this phase.  Join our [Discord server](https://www.manim.community/discord/) for updates and discussion.

Most developers use `uv` for package management. Install it and consult the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html).

## How to Cite Manim

To properly acknowledge the value of Manim in your research, cite the repository using the "cite this repository" button on the [GitHub page](https://github.com/ManimCommunity/manim), which will generate a citation in your preferred format.

## Code of Conduct

Our Code of Conduct, and how it's enforced, is available on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is dual-licensed under the MIT license. Copyright is held by both 3blue1brown LLC (see `LICENSE`) and the Manim Community Developers (see `LICENSE.community`).