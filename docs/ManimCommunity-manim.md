<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png"></a>
</p>

# Manim: An Open-Source Animation Engine for Mathematical Explanations

Create stunning mathematical visualizations and explanatory videos with Manim, a powerful Python library.  [View the original repository](https://github.com/ManimCommunity/manim).

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

Manim, the open-source animation engine, empowers you to create visually appealing and informative math videos, just like those by 3Blue1Brown!

**Key Features:**

*   **Programmatic Animations:** Define animations with Python code for precise control and repeatability.
*   **Mathematical Visualization:** Create complex mathematical objects and transformations with ease.
*   **Versatile Scene Creation:** Build scenes from simple to complex, tailored to your specific needs.
*   **Community-Driven:** Benefit from an active community, ensuring ongoing development and support.
*   **Interactive Tutorials:** Get started quickly with Jupyter Notebook integration.
*   **Docker Support:** Easily set up your environment with a ready-to-use Docker image.

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
> These instructions are for the community version _only_. Trying to use these instructions to install [3b1b/manim](https://github.com/3b1b/manim) or instructions there to install this version will cause problems. Read [this](https://docs.manim.community/en/stable/faq/installation.html#why-are-there-different-versions-of-manim) and decide which version you wish to install, then only follow the instructions for your desired version.

Manim has dependencies that you must install before you use it. Try it out [in our online Jupyter environment](https://try.manim.community/) before installing locally.

For local installation, please visit the [Documentation](https://docs.manim.community/en/stable/installation.html)
and follow the appropriate instructions for your operating system.

## Usage

Manim is an extremely versatile package. The following is an example `Scene` you can construct:

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

In order to view the output of this scene, save the code in a file called `example.py`. Then, run the following in a terminal window:

```sh
manim -p -ql example.py SquareToCircle
```

You should see your native video player program pop up and play a simple scene in which a square is transformed into a circle. You may find some more simple examples within this
[GitHub repository](example_scenes). You can also visit the [official gallery](https://docs.manim.community/en/stable/examples.html) for more advanced examples.

Manim also ships with a `%%manim` IPython magic which allows to use it conveniently in JupyterLab (as well as classic Jupyter) notebooks. See the
[corresponding documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) for some guidance and
[try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command line arguments

The general usage of Manim is as follows:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag in the command above is for previewing, meaning the video file will automatically open when it is done rendering. The `-ql` flag is for a faster rendering at a lower quality.

Some other useful flags include:

- `-s` to skip to the end and just show the final frame.
- `-n <number>` to skip ahead to the `n`'th animation of a scene.
- `-f` show the file in the file browser.

For a thorough list of command line arguments, visit the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

The community maintains a Docker image (`manimcommunity/manim`), available on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Find installation and usage instructions in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Get help with Manim on our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/). Report bugs or request features by opening an issue.

## Contributing

Contributions to Manim are welcome, especially for tests and documentation. See the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines.

Note:  Manim is currently undergoing a major refactor. New feature contributions will generally be paused during this period. Join our [Discord server](https://www.manim.community/discord/) for the latest updates and to discuss potential contributions.

Most developers on the project use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

Please cite Manim in your work to demonstrate its value. The best way is to use the "cite this repository" button on the [repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Read our full code of conduct and enforcement details on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

The software is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).