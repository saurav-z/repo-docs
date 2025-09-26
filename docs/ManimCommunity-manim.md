<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
</p>
<p align="center">
  <i><b>Bring your math and science explanations to life with stunning animations using Manim, the open-source animation engine.</b></i>
</p>
<br />

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

Manim is a powerful Python library that empowers you to create high-quality, mathematical animations programmatically, similar to those seen in the educational videos of [3Blue1Brown](https://www.3blue1brown.com/). This community-driven version, [ManimCE](https://github.com/ManimCommunity/manim), provides enhanced features, active community support, and continuous development.

**Key Features:**

*   **Programmatic Animation:** Create precise and customizable animations using Python code.
*   **Mathematical Visualization:** Ideal for illustrating mathematical concepts, scientific principles, and educational content.
*   **Community-Driven:** Benefit from an active community, regular updates, and extensive documentation.
*   **Versatile Output:** Generate videos, GIFs, and other formats.
*   **Jupyter Notebook Integration:** Seamlessly integrate Manim into Jupyter notebooks using the `%%manim` magic command.
*   **Docker Support:** Easily set up and run Manim using Docker containers.

## Table of Contents

*   [Installation](#installation)
*   [Usage](#usage)
*   [Command Line Arguments](#command-line-arguments)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help and Community](#help-with-manim)
*   [Contributing](#contributing)
*   [How to Cite Manim](#how-to-cite-manim)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

## Installation

**Important:** These instructions are for the Manim Community Edition (ManimCE) only.  For detailed installation instructions tailored to your operating system, please consult the official [documentation](https://docs.manim.community/en/stable/installation.html). If you would like to study how Grant Sanderson makes his videos, head over to his repository ([3b1b/manim](https://github.com/3b1b/manim)).

If you want to try it out first before installing it locally, you can do so [in our online Jupyter environment](https://try.manim.community/).

## Usage

Create visually stunning animations by writing Python code. Here is a simple example:

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

Save this code as `example.py` and then run the following command in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will render and preview a video of a square transforming into a circle. Explore more examples within the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html) to discover more advanced animation techniques.

Manim also offers a `%%manim` IPython magic for use in JupyterLab (and classic Jupyter) notebooks.  Refer to the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) for guidance and [try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

Manim uses the following command structure:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the output automatically. The `-ql` flag renders at a lower quality for faster rendering. Some other useful flags include:

-   `-s` to skip to the end and just show the final frame.
-   `-n <number>` to skip ahead to the `n`'th animation of a scene.
-   `-f` show the file in the file browser.

For a complete list of command-line arguments, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Manim is readily available as a Docker image, `manimcommunity/manim`, which can be found [on DockerHub](https://hub.docker.com/r/manimcommunity/manim). Instructions on how to install and use it are in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help and Community

Get help and connect with the Manim community on our [Discord Server](https://www.manim.community/discord/) or the [Reddit Community](https://www.reddit.com/r/manim/). For bug reports or feature requests, please open an issue.

## Contributing

Contributions to Manim are highly welcome! Please see the [documentation](https://docs.manim.community/en/stable/contributing.html) for the contribution guidelines. Consider joining our [Discord server](https://www.manim.community/discord/) for discussions and updates, especially considering the ongoing refactor.  The [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) provides instructions on installing with `uv`.

## How to Cite Manim

If you use Manim in your work, please cite the project by using the "cite this repository" button on the [repository page](https://github.com/ManimCommunity/manim) to generate a citation in your preferred format.

## Code of Conduct

Review the full Code of Conduct and enforcement details on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

The software is double-licensed under the MIT license (copyright by 3blue1brown LLC and Manim Community Developers). See the [LICENSE](LICENSE) files for details.

[**Back to Top**](https://github.com/ManimCommunity/manim#readme)