<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
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

# Manim: Create Stunning Math Animations with Python

**Manim is a powerful and versatile Python library that empowers you to generate impressive mathematical visualizations and educational videos.**

**[Check out the original repo](https://github.com/ManimCommunity/manim)**

**Key Features:**

*   **Precise Animations:** Create animations programmatically, ensuring accuracy and control.
*   **Python-Based:** Leverage the flexibility and power of Python for your animations.
*   **Community-Driven:** Benefit from an active community, regular updates, and improved features.
*   **Versatile Applications:** Ideal for creating educational content, mathematical explanations, and visual storytelling.
*   **Cross-Platform Compatibility:** Available on various platforms, including Windows, macOS, and Linux.
*   **Interactive Development:** Experiment easily with scenes in Jupyter notebooks.

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

**Note:** These instructions are for the community version only. Refer to the [Documentation](https://docs.manim.community/en/stable/installation.html) for detailed installation guides specific to your operating system. If you're new to Manim, you can also try it out first in [our online Jupyter environment](https://try.manim.community/).

## Usage

Here's a basic example demonstrating how to create a square-to-circle animation:

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

1.  Save the code in a file (e.g., `example.py`).
2.  Run the following command in your terminal:

    ```bash
    manim -p -ql example.py SquareToCircle
    ```

This command will generate and display a video of the square transforming into a circle.  Explore more examples in the [GitHub repository](example_scenes) or the [official gallery](https://docs.manim.community/en/stable/examples.html).

You can also use Manim within JupyterLab notebooks.  See the [corresponding documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) or [try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

Manim's general usage:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

*   `-p`: Preview the video automatically.
*   `-ql`: Render at a lower quality (faster).
*   `-s`: Show the final frame only.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Show the output file in the file browser.

For a comprehensive list of arguments, refer to the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Manim also provides a Docker image (`manimcommunity/manim`) on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Find installation and usage instructions in our [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

For assistance with installation or usage, join our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs or suggest features by opening an issue on GitHub.

## Contributing

Contributions are welcome, especially for tests and documentation.  See the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines.

**Important:** The project is undergoing a major refactor, so new feature contributions may not be accepted currently.  Join our [Discord server](https://www.manim.community/discord/) for the latest developments and to discuss potential contributions.  We use `uv` for management. Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

If you use Manim in your work, please cite the repository. You can generate a citation in your preferred format by clicking the "cite this repository" button on the [GitHub page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Read our full code of conduct and enforcement policies on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

The software is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).