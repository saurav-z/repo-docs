<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
</p>

<h1 align="center">Manim: An Animation Engine for Explanatory Math Videos</h1>

**Create stunning math animations and educational videos with Manim, the open-source Python library inspired by 3Blue1Brown.** ([Original Repo](https://github.com/ManimCommunity/manim))

<br/>

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
[![CI](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)

<hr/>

Manim is a powerful animation engine enabling you to create visually engaging mathematical explanations. This community-driven version, ManimCE, offers enhanced features, active development, and a thriving community, making it the ideal choice for your animation needs.

**Key Features:**

*   **Programmatic Animation:** Define animations using Python code for precise control.
*   **Versatile Scene Creation:** Build complex scenes with geometric objects, text, and mathematical expressions.
*   **Customizable Visuals:** Tailor animations with a wide range of colors, styles, and effects.
*   **Integration with Jupyter Notebooks:** Utilize Manim directly within Jupyter environments for interactive development.
*   **Community Driven:** Benefit from active community support, extensive documentation, and ongoing improvements.
*   **Docker Support:** Easily set up and run Manim using Docker containers.

## Table of Contents

*   [Installation](#installation)
*   [Usage](#usage)
*   [Command Line Arguments](#command-line-arguments)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help and Community](#help-with-manim)
*   [Contributing](#contributing)
*   [How to Cite](#how-to-cite)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

## Installation

To get started with Manim, follow the installation instructions provided in the [documentation](https://docs.manim.community/en/stable/installation.html).  For a quick try, check out the [online Jupyter environment](https://try.manim.community/).

## Usage

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

Save this code as `example.py` and run it using the command line:

```bash
manim -p -ql example.py SquareToCircle
```

This will generate a video transforming a square into a circle. Explore the [official gallery](https://docs.manim.community/en/stable/examples.html) and the [GitHub repository](example_scenes) for more examples.

Manim also integrates with JupyterLab and Jupyter notebooks using the `%%manim` magic command.  See the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and the [online example](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

The general structure for using Manim from the command line is:

```bash
manim [options] [file.py] [SceneName]
```

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

Common arguments include:

*   `-p`: Preview the video automatically.
*   `-ql`: Render at a lower quality for faster results.
*   `-s`: Show the final frame only.
*   `-n <number>`: Skip to the `n`th animation.
*   `-f`: Open the output file in the file browser.

For a complete list of command-line arguments, consult the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Use the official Docker image (`manimcommunity/manim`) for easy setup and consistent environments. Find it on [DockerHub](https://hub.docker.com/r/manimcommunity/manim), with instructions in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help and Community

Join the vibrant Manim community for support and discussions:

*   [Discord Server](https://www.manim.community/discord/)
*   [Reddit Community](https://www.reddit.com/r/manim/)

Report bugs or request features by opening an issue.

## Contributing

Contributions are welcome! Please review the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html).  Stay informed about ongoing refactoring efforts, and discuss potential contributions on our [Discord server](https://www.manim.community/discord/).

Manim developers use `uv` for environment management; see its [documentation](https://docs.astral.sh/uv/) and the [Manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html).

## How to Cite

Cite Manim in your research by using the "cite this repository" button on the [repository page](https://github.com/ManimCommunity/manim). This generates a citation in your preferred format.

## Code of Conduct

Read our full code of conduct at [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is dual-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and Manim Community Developers (see LICENSE.community).