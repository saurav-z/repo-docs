<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png"></a>
</p>

# Manim: Create Stunning Math Animations (Community Edition)

**Bring your mathematical ideas to life with Manim, a powerful and versatile animation engine for generating explanatory math videos.** This community-driven version of Manim builds upon the foundations laid by 3Blue1Brown, offering enhanced features, improved documentation, and a vibrant community to support your animation journey.  [See the original Manim repository](https://github.com/ManimCommunity/manim) for more information.

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
[![CI](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)

## Key Features:

*   **Precise Animations:** Programmatically create complex mathematical animations with fine-grained control.
*   **Python-Based:** Leverage the power and flexibility of Python for scripting your scenes.
*   **Community-Driven:** Benefit from active community development, feature enhancements, and extensive documentation.
*   **Jupyter Integration:** Seamlessly integrate Manim into Jupyter notebooks for interactive experimentation.
*   **Docker Support:** Easily set up and run Manim using Docker containers.

## Quick Start

Here's a simple example to get you started:

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

Save the code as `example.py` and run:

```bash
manim -p -ql example.py SquareToCircle
```

This will generate a video transforming a square into a circle.

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

For detailed installation instructions, please refer to the [Installation Guide](https://docs.manim.community/en/stable/installation.html) in the official documentation. You can also try Manim in an [online Jupyter environment](https://try.manim.community/) without installing anything.

## Usage

Manim is designed for creating mathematical animations. See the example in the [Quick Start](#quick-start) section.

Explore more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also offers an `%%manim` IPython magic for easy use in JupyterLab and Jupyter notebooks. Consult the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and [try it online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

Use the command-line to control Manim's behavior.

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

Common flags:
*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality for faster processing.
*   `-s`: Show the final frame only.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Show the file in the file browser.

For a complete list, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

The Manim Community provides a Docker image (`manimcommunity/manim`) on [DockerHub](https://hub.docker.com/r/manimcommunity/manim). Learn how to use it in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Get assistance and connect with the community via the [Discord Server](https://www.manim.community/discord/) or the [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs and request features by opening an issue.

## Contributing

Contributions are welcome! Join the [Discord server](https://www.manim.community/discord/) to discuss potential contributions and stay informed. See the [contribution guide](https://docs.manim.community/en/stable/contributing.html) for details.
Remember to install `uv` and utilize it to install manim according to the instructions at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html).

## How to Cite Manim

To cite Manim in your work, please use the "cite this repository" feature on the [GitHub repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Read our Code of Conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is dual-licensed under the [MIT license](LICENSE) with copyright held by 3blue1brown LLC and the Manim Community Developers.