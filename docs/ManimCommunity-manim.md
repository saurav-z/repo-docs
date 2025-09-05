<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png"></a>
    <br />
    <br />
</p>

# Manim: Create Stunning Math Animations

**Manim is a powerful Python library that empowers you to generate precise, visually appealing animations for mathematical explanations and educational videos.** Build engaging content with Manim and take your audience on a captivating journey through complex concepts.  Explore the original repo [here](https://github.com/ManimCommunity/manim).

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
[![CI](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)](https://github.com/ManimCommunity/manim/workflows/CI)

## Key Features

*   **Precise Animations:** Create animations with mathematical precision, ideal for explaining complex concepts.
*   **Programmatic Control:**  Define animations through Python code, allowing for flexibility and customization.
*   **Community-Driven:** Benefit from active community support, continuous development, and improved features.
*   **Versatile Applications:** Perfect for educational videos, presentations, and visual explanations of mathematics and other fields.
*   **Jupyter Notebook Integration:** Seamlessly integrate Manim into your JupyterLab and Jupyter notebooks for interactive animation creation.
*   **Docker Support**: Easy-to-use Docker image to get started.

## Table of Contents

-   [Installation](#installation)
-   [Usage](#usage)
-   [Command Line Arguments](#command-line-arguments)
-   [Documentation](#documentation)
-   [Docker](#docker)
-   [Help and Community](#help-with-manim)
-   [Contributing](#contributing)
-   [How to Cite Manim](#how-to-cite-manim)
-   [Code of Conduct](#code-of-conduct)
-   [License](#license)

## Installation

> [!CAUTION]
> These instructions are for the community version _only_. Read [this](https://docs.manim.community/en/stable/faq/installation.html#why-are-there-different-versions-of-manim) and decide which version you wish to install, then only follow the instructions for your desired version.

Manim has dependencies you'll need to install.  You can try it out online via [our online Jupyter environment](https://try.manim.community/)

For local installation, visit the [Documentation](https://docs.manim.community/en/stable/installation.html) and follow the instructions for your operating system.

## Usage

Manim is versatile. Example `Scene` construction:

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

Save the code in `example.py` and run:

```bash
manim -p -ql example.py SquareToCircle
```

You'll see a square transform into a circle. Examples are in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also has a `%%manim` IPython magic for use in JupyterLab (as well as classic Jupyter) notebooks, with documentation [here](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and an online [try it out environment](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

General Manim usage:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the video; `-ql` renders faster at a lower quality.

Other useful flags:

*   `-s`: Show final frame.
*   `-n <number>`: Skip to animation `n`.
*   `-f`: Show file in file browser.

Full command line arguments are in the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Find the documentation at [ReadTheDocs](https://docs.manim.community/).

## Docker

The community maintains a Docker image (`manimcommunity/manim`) on [DockerHub](https://hub.docker.com/r/manimcommunity/manim). Instructions are in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help and Community

For help, reach out to the [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/). For bug reports/feature requests, open an issue.

## Contributing

Contributions are welcome!  Tests and documentation are needed.  See the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines.

Manim is undergoing a major refactor, so contributions implementing new features may not be accepted during this period.  Join our [Discord server](https://www.manim.community/discord/) for the latest updates.

Most developers use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

Cite Manim to show its value in your research.  Go to the [repository page](https://github.com/ManimCommunity/manim) and click the "cite this repository" button on the right sidebar to generate a citation in your preferred format.

## Code of Conduct

Read the full code of conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).