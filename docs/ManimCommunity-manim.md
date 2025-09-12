# Manim: Create Stunning Math Animations with Python

**Bring your mathematical explanations to life** with Manim, a powerful Python library for generating precise and visually engaging animations, perfect for educational videos and presentations. Explore the original repository on [GitHub](https://github.com/ManimCommunity/manim).

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

---

Manim is the animation engine behind the popular math educational videos of [3Blue1Brown](https://www.3blue1brown.com/), allowing you to create engaging and informative visuals.

**Key Features:**

*   **Precise Animations:** Control every aspect of your animations with programmatic precision.
*   **Versatile:** Create a wide variety of visuals, from simple diagrams to complex mathematical concepts.
*   **Python-Based:** Leverage the power and flexibility of Python to define your animations.
*   **Community-Driven:** Benefit from an active and supportive community contributing to the project's ongoing development.
*   **JupyterLab Integration:** Seamlessly use Manim within JupyterLab notebooks for interactive exploration.
*   **Docker Support:** Utilize Docker for easy installation and environment management.

## Table of Contents:

-   [Installation](#installation)
-   [Usage](#usage)
-   [Command Line Arguments](#command-line-arguments)
-   [Documentation](#documentation)
-   [Docker](#docker)
-   [Help with Manim](#help-with-manim)
-   [Contributing](#contributing)
-   [How to Cite Manim](#how-to-cite-manim)
-   [Code of Conduct](#code-of-conduct)
-   [License](#license)

## Installation

> [!CAUTION]
> These instructions are for the community version _only_.  Ensure you are installing the correct version.

To install Manim locally, follow the detailed instructions in the [Documentation](https://docs.manim.community/en/stable/installation.html) for your operating system.  Alternatively, try Manim directly in your browser using the [online Jupyter environment](https://try.manim.community/).

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

Save this code as `example.py` and run it from your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

Explore more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also works within JupyterLab notebooks. Refer to the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and [try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

The general structure for running Manim is:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

For example, `-p` previews the video, and `-ql` renders at a lower quality.

Some helpful flags:

*   `-s`:  Show only the final frame.
*   `-n <number>`: Skip to the *n*th animation of a scene.
*   `-f`: Show the output file in the file browser.

See the [documentation](https://docs.manim.community/en/stable/guides/configuration.html) for a full list.

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

The Manim Community provides a Docker image (`manimcommunity/manim`) on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Find installation and usage instructions in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Get help and connect with the community on our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/). Report bugs and request features by opening an issue.

## Contributing

Contributions are highly welcome! For guidelines, see the [documentation](https://docs.manim.community/en/stable/contributing.html).

Please note:  Manim is currently undergoing a major refactor. Join our [Discord server](https://www.manim.community/discord/) to discuss potential contributions and stay updated on developments.

Most developers use `uv` for management. See the [documentation](https://docs.astral.sh/uv/) for more information about `uv`, and [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) for install manim with uv.

## How to Cite Manim

To support the value of Manim, please cite it in your work. Generate a citation from the [repository page](https://github.com/ManimCommunity/manim) using the "cite this repository" button.

## Code of Conduct

Read our full code of conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and Manim Community Developers (see LICENSE.community).