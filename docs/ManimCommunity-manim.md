<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
    <br />
    <br />
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
    <br />
</p>

# Manim: Create Stunning Math Animations with Code

**Manim is a powerful and versatile Python library that empowers you to generate beautiful, precise animations for mathematical explanations and educational videos.**  For the original repo, see:  [ManimCommunity/manim](https://github.com/ManimCommunity/manim)

**Key Features:**

*   **Programmatic Animation:** Define animations using Python code for complete control and precision.
*   **Mathematical Visualization:** Easily create and animate mathematical objects, formulas, and concepts.
*   **Community-Driven:** Benefit from an active community, continuous development, and comprehensive documentation.
*   **Versatile Output:** Generate videos in various formats and resolutions.
*   **Jupyter Integration:** Utilize the `%%manim` IPython magic for seamless animation creation within Jupyter notebooks.

## Table of Contents

*   [Installation](#installation)
*   [Usage](#usage)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help with Manim](#help-with-manim)
*   [Contributing](#contributing)
*   [How to Cite Manim](#how-to-cite-manim)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

## Installation

ManimCE requires a few dependencies. If you'd like to try it out before installing locally, you can do so [in our online Jupyter environment](https://try.manim.community/).

For local installation, please visit the [Documentation](https://docs.manim.community/en/stable/installation.html) and follow the appropriate instructions for your operating system.

## Usage

Manim is a very versatile package. Here's an example `Scene`:

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

To view the output of this scene, save the code to a file called `example.py`. Then, run the following in a terminal window:

```sh
manim -p -ql example.py SquareToCircle
```

You should see a video player program pop up, showing a simple scene where a square transforms into a circle. Explore more examples within this [GitHub repository](example_scenes) or the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also ships with a `%%manim` IPython magic that allows you to use it conveniently in JupyterLab (and classic Jupyter) notebooks. See the [corresponding documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and [try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

The general usage of Manim is as follows:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag is for previewing.  `-ql` is for faster, lower-quality rendering.

Some other useful flags:

*   `-s`: Skip to the final frame.
*   `-n <number>`: Skip ahead to the `n`th animation.
*   `-f`: Show the file in the file browser.

For a thorough list of command line arguments, visit the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

A Docker image (`manimcommunity/manim`) is maintained on [DockerHub](https://hub.docker.com/r/manimcommunity/manim). Instructions on how to install and use it can be found in our [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Get help with installation and usage on our [Discord
Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim). Submit bug reports or feature requests by opening an issue.

## Contributing

Contributions are welcome! There is a dire need for tests and documentation. For contribution guidelines, see the [documentation](https://docs.manim.community/en/stable/contributing.html).

Please note that Manim is undergoing a major refactor, so contributions for new features are generally not accepted at this time. Join our
[Discord server](https://www.manim.community/discord/) to discuss potential contributions and keep up to date.

Most developers on the project use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

Cite Manim in your work by going to our
[repository page](https://github.com/ManimCommunity/manim) and clicking the "cite this repository" button.

## Code of Conduct

Read our full code of conduct, and how we enforce it, on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

The software is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).