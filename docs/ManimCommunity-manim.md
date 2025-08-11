<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
</p>

<h1 align="center">Manim: Create Stunning Math Animations</h1>

<p align="center">
    <em>Bring your mathematical concepts to life with the power of Python!</em>
</p>

<p align="center">
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

<hr />

Manim is a powerful Python library for generating high-quality mathematical animations. Perfect for educators, researchers, and anyone looking to visualize complex concepts, Manim allows you to create precise and visually engaging videos programmatically.  Inspired by the work of [3Blue1Brown](https://www.3blue1brown.com/), Manim provides a flexible and intuitive framework for bringing your ideas to life.  This is the Community Edition (ManimCE), actively maintained and developed by the community.  For the original project, see [3b1b/manim](https://github.com/3b1b/manim).

**Key Features:**

*   **Programmatic Animation:** Create animations with Python code, giving you complete control over every detail.
*   **Precise Graphics:** Produce sharp, clear visuals with vector graphics.
*   **Mathematical Formulas:** Seamlessly integrate LaTeX for displaying equations and mathematical notation.
*   **Scene Composition:**  Build complex animations by combining various scenes and effects.
*   **Customizable:** Fine-tune animations with a wide array of options for colors, styles, and transitions.
*   **Community-Driven:** Benefit from an active community, extensive documentation, and continuous improvements.

## Table of Contents

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

> [!CAUTION]
> These instructions are for the community version _only_. Trying to use these instructions to install [3b1b/manim](https://github.com/3b1b/manim) or instructions there to install this version will cause problems. Read [this](https://docs.manim.community/en/stable/faq/installation.html#why-are-there-different-versions-of-manim) and decide which version you wish to install, then only follow the instructions for your desired version.

Before you begin, ensure you've installed the necessary dependencies. Try out Manim directly via our [online Jupyter environment](https://try.manim.community/)!

For detailed installation instructions tailored to your operating system, consult the official [Manim Documentation](https://docs.manim.community/en/stable/installation.html).

## Usage

Manim's versatility allows you to craft a wide range of animations.  Here's a simple example to get you started:

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

```bash
manim -p -ql example.py SquareToCircle
```

This will generate and display a video transforming a square into a circle.  Explore more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also integrates seamlessly with JupyterLab (and classic Jupyter) notebooks using the `%%manim` IPython magic. Learn more in the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and [try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

Manim's command-line interface offers a range of options for customizing your animations.

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

Here's what the command in the example means:

*   `-p`: Preview the video automatically.
*   `-ql`: Render at a lower quality for faster results.

Other helpful flags include:

*   `-s`: Show only the final frame.
*   `-n <number>`: Skip to the nth animation in a scene.
*   `-f`: Open the output file in your file browser.

For an exhaustive list of command-line arguments, please consult the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

We provide a Docker image (`manimcommunity/manim`) available on [DockerHub](https://hub.docker.com/r/manimcommunity/manim). Installation and usage instructions can be found in our [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Need assistance with installation or usage?  Reach out to our friendly community on our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/).  For bug reports or feature requests, please open an issue.

## Contributing

Contributions to Manim are greatly appreciated. The project has a dire need for tests and documentation. Please consult the [documentation](https://docs.manim.community/en/stable/contributing.html) for contribution guidelines.

Please note that Manim is currently undergoing a significant refactor. During this period, contributions implementing new features may not be accepted. Join our [Discord server](https://www.manim.community/discord/) for the latest updates and to discuss potential contributions.

Most developers use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

To support research and demonstrate the value of Manim, please cite it in your work. Please go to our [repository page](https://github.com/ManimCommunity/manim) and click the "cite this repository" button on the right sidebar. This will generate a citation in your preferred format.

## Code of Conduct

Our full code of conduct, and how we enforce it, can be read on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).