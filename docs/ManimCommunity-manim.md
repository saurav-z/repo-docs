<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
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
</p>

# Manim: Create Stunning Math Animations with Python

**Manim, the animation engine for explanatory math videos, empowers you to bring mathematical concepts to life with stunning visuals.** ([Original Repo](https://github.com/ManimCommunity/manim))

## Key Features:

*   **Programmatic Animation:** Define animations precisely using Python code.
*   **Versatile Scene Construction:** Build complex scenes with geometric shapes, text, and mathematical objects.
*   **Customizable Visuals:** Control colors, styles, and transformations to create unique animations.
*   **Command-Line Tools:** Easily render and manage your animations with a powerful command-line interface.
*   **Jupyter Notebook Integration:** Seamlessly integrate Manim into your JupyterLab notebooks.
*   **Community-Driven:** Benefit from an active and supportive community, documentation, and resources.

## Table of Contents:

*   [Installation](#installation)
*   [Usage](#usage)
*   [Command Line Arguments](#command-line-arguments)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help with Manim](#help-with-manim)
*   [Contributing](#contributing)
*   [How to Cite Manim](#how-to-cite-manim)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

## Installation

Manim requires dependencies; install following the instructions for your [operating system](https://docs.manim.community/en/stable/installation.html). Alternatively, try Manim [in our online Jupyter environment](https://try.manim.community/).

## Usage

Here's a simple example of a `Scene`:

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

Save this as `example.py` and run:

```bash
manim -p -ql example.py SquareToCircle
```

View more [examples](https://docs.manim.community/en/stable/examples.html). Manim also works with `%%manim` in Jupyter notebooks.

## Command Line Arguments

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

Key arguments:

*   `-p`: Preview video.
*   `-ql`: Faster rendering at a lower quality.
*   `-s`: Show the final frame.
*   `-n <number>`: Skip to the `n`th animation.
*   `-f`: Show the file in the file browser.

See the [documentation](https://docs.manim.community/en/stable/guides/configuration.html) for a full list.

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Use the official [Docker image](https://hub.docker.com/r/manimcommunity/manim) with instructions in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Join the [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/) for help. Submit bug reports and feature requests by opening an issue on [GitHub](https://github.com/ManimCommunity/manim).

## Contributing

Contributions are welcome! Consult the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html). Join the [Discord server](https://www.manim.community/discord/) to discuss contributions.

## How to Cite Manim

Cite Manim from the [repository page](https://github.com/ManimCommunity/manim) by clicking the "cite this repository" button.

## Code of Conduct

Read our [Code of Conduct](https://docs.manim.community/en/stable/conduct.html).

## License

Licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and Manim Community Developers (see LICENSE.community).