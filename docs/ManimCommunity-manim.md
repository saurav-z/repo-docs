<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png"></a>
    <br />
    <br />
    <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
    <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit" href=></a>
    <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter">
    <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"> </a>
    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
    <br />
    <br />
</p>

# Manim: Animate Your Math Concepts with Stunning Visuals

Manim is a powerful Python library that lets you create visually stunning mathematical animations, perfect for explaining complex concepts.  **[Check out the original repository](https://github.com/ManimCommunity/manim).**

## Key Features:

*   **Precise Programmatic Control:** Create animations with fine-grained control over every detail.
*   **Versatile Scene Construction:**  Build complex scenes using a flexible and intuitive API.
*   **Community Driven:** Benefit from active development, improved features, and extensive documentation.
*   **Jupyter Notebook Integration:** Utilize `%%manim` magic for seamless animation creation within Jupyter environments.
*   **Cross-Platform Support:** Works on various operating systems with comprehensive installation instructions.
*   **Docker Image Available:**  Easily set up your environment with the official Docker image.

## Table of Contents:

-   [Installation](#installation)
-   [Usage](#usage)
-   [Documentation](#documentation)
-   [Docker](#docker)
-   [Help with Manim](#help-with-manim)
-   [Contributing](#contributing)
-   [How to Cite Manim](#how-to-cite-manim)
-   [Code of Conduct](#code-of-conduct)
-   [License](#license)

## Installation

> [!CAUTION]
> These instructions are for the community version _only_.  Follow the instructions for your desired version.

Manim requires dependencies that must be installed before use.  Try it out first [in our online Jupyter environment](https://try.manim.community/). For local installation, please visit the [Documentation](https://docs.manim.community/en/stable/installation.html).

## Usage

Example `Scene`:

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

Save the code to `example.py` and run:

```sh
manim -p -ql example.py SquareToCircle
```

Find more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also features a `%%manim` IPython magic for use in JupyterLab (and classic Jupyter) notebooks. See the [corresponding documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) or [try it online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command line arguments

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews, `-ql` renders at lower quality.

Other useful flags:
*   `-s`: skip to the end.
*   `-n <number>`: skip to animation `n`.
*   `-f`: show the file in the file browser.

For a thorough list, visit the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Documentation is at [ReadTheDocs](https://docs.manim.community/).

## Docker

The community maintains a Docker image (`manimcommunity/manim`) [on DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Find install/usage instructions in our [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Get help on our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim).  Submit bug reports or feature requests by opening an issue.

## Contributing

Contributions are welcome!  See the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines. Note: New feature contributions are paused during refactor. Join our [Discord server](https://www.manim.community/discord/) for updates.

Developers use `uv` for management; ensure `uv` is installed. Learn more at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

Cite Manim by clicking the "cite this repository" button on the [repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Read our code of conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and Manim Community Developers (see LICENSE.community).