<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
    <br />
    <br />
</p>

# Manim: Create Stunning Math Animations with Python

**Bring your mathematical concepts to life with Manim, an open-source Python animation engine perfect for creating engaging and visually appealing explanatory videos.** Explore the full potential of Manim and contribute to its community!

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

<hr />

Manim is an open-source Python library for creating mathematical animations. It empowers you to visualize complex concepts and create compelling educational content. This version, Manim Community Edition (ManimCE), is actively maintained and developed by the community. Check out the original Manim project by 3Blue1Brown: [3b1b/manim](https://github.com/3b1b/manim).

## Key Features

*   **Precise Animations:** Create animations with pixel-perfect accuracy, ideal for illustrating mathematical principles.
*   **Programmatic Control:** Define animations using Python code, offering flexibility and customization.
*   **Versatile Scene Design:** Construct scenes with various 2D and 3D objects, transformations, and text elements.
*   **Community-Driven Development:** Benefit from an active community, improved documentation, and ongoing feature enhancements.
*   **Interactive Environment:** Utilize Manim within Jupyter notebooks with the `%%manim` magic command for interactive experimentation.

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
> These instructions are for the community version _only_. Trying to use these instructions to install [3b1b/manim](https://github.com/3b1b/manim) or instructions there to install this version will cause problems. Read [this](https://docs.manim.community/en/stable/faq/installation.html#why-are-there-different-versions-of-manim) and decide which version you wish to install, then only follow the instructions for your desired version.

Before installing Manim locally, ensure that necessary dependencies are in place. You can try Manim without installing it by using our [online Jupyter environment](https://try.manim.community/).

For local installation, follow the instructions for your operating system on the [Documentation](https://docs.manim.community/en/stable/installation.html).

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

Save the code in a file named `example.py` and run the following command in your terminal:

```sh
manim -p -ql example.py SquareToCircle
```

This command will render a video showing a square transforming into a circle. Explore the [example_scenes](https://github.com/ManimCommunity/manim/tree/main/example_scenes) within this repository for further examples, or visit the [official gallery](https://docs.manim.community/en/stable/examples.html) for more advanced demonstrations.

Manim also integrates with JupyterLab through the `%%manim` IPython magic, allowing interactive use in Jupyter notebooks. See the [corresponding documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and try it out online [here](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

The general usage of Manim is as follows:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the video, and `-ql` renders at a lower quality for faster processing.

Additional flags:

*   `-s`: Shows the final frame.
*   `-n <number>`: Skips to the `n`th animation.
*   `-f`: Opens the file in the file browser.

Refer to the [documentation](https://docs.manim.community/en/stable/guides/configuration.html) for a complete list of command-line arguments.

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

The Manim community maintains a Docker image (`manimcommunity/manim`) on [DockerHub](https://hub.docker.com/r/manimcommunity/manim). Learn how to install and use the Docker image in our [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

For assistance with installation or usage, reach out to our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/). Submit bug reports and feature requests by opening an issue.

## Contributing

We welcome contributions to Manim. We especially need help with tests and documentation. See the [documentation](https://docs.manim.community/en/stable/contributing.html) for contribution guidelines.

Please note that Manim is currently undergoing a major refactor, and contributions for new features may not be accepted. Join our [Discord server](https://www.manim.community/discord/) to stay updated and discuss potential contributions.

Developers use `uv` for dependency management. Please install it and then learn how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

To support the value of Manim and ensure your work is properly attributed, please cite Manim in your publications using the "cite this repository" button on the [repository page](https://github.com/ManimCommunity/manim). This will generate a citation in your preferred format.

## Code of Conduct

Our code of conduct is detailed on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

The software is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).