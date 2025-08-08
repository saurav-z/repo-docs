<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
    <br />
    <br />
</p>

# Manim: Create Stunning Math Animations with Code

**Transform complex mathematical concepts into captivating visuals with Manim, the open-source animation engine.**  Based on the work of 3Blue1Brown, Manim empowers you to create clear and engaging explanatory math videos through code.  Get started today by visiting the [original repo](https://github.com/ManimCommunity/manim).

[![PyPI Latest Release](https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi)](https://pypi.org/project/manim/)
[![Docker image](https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker)](https://hub.docker.com/r/manimcommunity/manim)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
[![MIT License](https://img.shields.io/badge/license-MIT-red.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit)](https://www.reddit.com/r/manim/)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community)](https://twitter.com/manim_community/)
[![Discord](https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord)](https://www.manim.community/discord/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/manimce/badge/?version=latest)](https://docs.manim.community/)
[![Downloads](https://pepy.tech/badge/manim/month?)](https://pepy.tech/project/manim)
[![CI](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)

<hr />

**Key Features:**

*   **Programmatic Animation:** Create animations with precise control using Python code.
*   **Versatile Scene Creation:** Build a wide variety of scenes, from simple diagrams to complex mathematical visualizations.
*   **Community-Driven Development:** Benefit from active community support, enhanced features, and ongoing improvements.
*   **Integration with Jupyter:** Use the `%%manim` IPython magic for convenient animation creation within Jupyter notebooks.
*   **Extensive Documentation:** Comprehensive documentation and examples to help you get started and master Manim.
*   **Docker Support:** Easily set up and run Manim using Docker containers.

## Table of Contents:

*   [Installation](#installation)
*   [Usage](#usage)
*   [Command line arguments](#command-line-arguments)
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

To install the Manim Community Edition, you'll need to set up a few dependencies.
For local installation, please visit the [Documentation](https://docs.manim.community/en/stable/installation.html)
and follow the appropriate instructions for your operating system.

If you'd like to test Manim without installing, you can try it out [in our online Jupyter environment](https://try.manim.community/).

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

Save this code as `example.py` and run the following command in your terminal:

```sh
manim -p -ql example.py SquareToCircle
```

This will render and preview a scene transforming a square into a circle. Explore more examples in the [GitHub repository](example_scenes) or the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also seamlessly integrates with Jupyter notebooks using the `%%manim` magic command.  See the [corresponding documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and [try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command line arguments

The general usage of Manim is as follows:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the video, while `-ql` renders at a lower quality.

Other useful flags:

*   `-s`: Show the final frame.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Show the file in the file browser.

For a complete list of command-line arguments, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Find comprehensive documentation at [ReadTheDocs](https://docs.manim.community/).

## Docker

The Manim Community also provides a Docker image (`manimcommunity/manim`) on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Find installation and usage instructions in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Need assistance?  Connect with the community on our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs or request features by opening an issue.

## Contributing

Contributions are always welcome!  We particularly need help with tests and documentation.  See the [documentation](https://docs.manim.community/en/stable/contributing.html) for contribution guidelines.

Please note that Manim is currently undergoing a major refactor. We highly recommend joining our
[Discord server](https://www.manim.community/discord/) to discuss any potential
contributions and keep up to date with the latest developments.

Most developers on the project use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

To properly credit Manim in your work, go to our [repository page](https://github.com/ManimCommunity/manim) and click the "cite this repository" button.

## Code of Conduct

Read our full code of conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).