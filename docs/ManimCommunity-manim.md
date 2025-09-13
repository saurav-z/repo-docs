<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
</p>

## Manim: Create Stunning Math Animations

**Manim is a powerful Python library for generating mathematical animations, perfect for creating educational videos and visualizations.** Based on the work of 3Blue1Brown, Manim empowers you to programmatically create precise and visually appealing animations.  Explore the original repository: [Manim on GitHub](https://github.com/ManimCommunity/manim).

**Key Features:**

*   **Precise Animations:** Create animations with pixel-perfect precision, ideal for illustrating mathematical concepts.
*   **Programmatic Control:**  Build animations through Python code, offering complete control over every aspect of the visualization.
*   **Versatile Scene Design:**  Construct complex scenes using a wide range of shapes, objects, and transformations.
*   **Community-Driven:** Benefit from a vibrant and active community that contributes to the library's development and provides support.
*   **Easy Integration:** Use Manim in Jupyter notebooks with the `%%manim` IPython magic.

**Key Highlights:**

*   **Latest Release:** [![PyPI Latest Release](https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi)](https://pypi.org/project/manim/)
*   **Docker Image:** [![Docker Image](https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker)](https://hub.docker.com/r/manimcommunity/manim)
*   **Try it Online:**  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
*   **License:** [![MIT License](https://img.shields.io/badge/license-MIT-red.svg?style=flat)](http://choosealicense.com/licenses/mit/)
*   **Community:** [![Reddit](https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit)](https://www.reddit.com/r/manim/)  [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community)](https://twitter.com/manim_community/) [![Discord](https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord)](https://www.manim.community/discord/)
*   **Code Style:** [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
*   **Documentation Status:** [![Documentation Status](https://readthedocs.org/projects/manimce/badge/?version=latest)](https://docs.manim.community/)
*   **Downloads:** [![Downloads](https://pepy.tech/badge/manim/month?)](https://pepy.tech/project/manim)
*   **CI:** [![CI](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)](https://github.com/ManimCommunity/manim/workflows/CI)

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

Manim has dependencies that must be installed prior to using it.  If you want to try it out before installing locally, use the [online Jupyter environment](https://try.manim.community/).

For local installation, follow the instructions in the [Documentation](https://docs.manim.community/en/stable/installation.html) for your operating system.

## Usage

Manim is a versatile package. Here's a basic example scene:

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

Save the code in a file named `example.py` and run:

```sh
manim -p -ql example.py SquareToCircle
```

This command will generate a simple animation transforming a square into a circle.  Find more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also supports IPython magic for use in JupyterLab (and classic Jupyter) notebooks. Refer to the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and [try it online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command line arguments

General Manim usage:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the video automatically; `-ql` renders at lower quality for speed.

Additional helpful flags:

*   `-s`: Shows the final frame.
*   `-n <number>`: Skips to the *n*th animation.
*   `-f`: Shows the file in the file browser.

For a complete list of command-line arguments, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

A Docker image (`manimcommunity/manim`) is available on [DockerHub](https://hub.docker.com/r/manimcommunity/manim). Installation and usage instructions are in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Get help with Manim on the [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs or suggest features by opening an issue.

## Contributing

Contributions are welcome.  Specifically, tests and documentation are needed. See the [documentation](https://docs.manim.community/en/stable/contributing.html) for contribution guidelines.

Please note that Manim is currently undergoing a major refactor. Contributions that implement new features will likely not be accepted.  Join the [Discord server](https://www.manim.community/discord/) to discuss potential contributions.

Most developers use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

Cite Manim in your work to demonstrate its value, generate a citation on the [repository page](https://github.com/ManimCommunity/manim) by clicking the "cite this repository" button.

## Code of Conduct

Read the full code of conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

This software is licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).