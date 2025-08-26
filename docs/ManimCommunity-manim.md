<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
    <br />
    <br />
    <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
    <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit" ></a>
    <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter">
    <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"> </a>
    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
    <br />
</p>

# Manim: Create Stunning Math Animations with Python

**Manim is a powerful animation engine for generating explanatory math videos, perfect for educators, researchers, and anyone wanting to visualize mathematical concepts.**  This community-driven version of the popular animation tool empowers you to create precise and visually engaging animations programmatically.  [Get started at the Manim Community GitHub](https://github.com/ManimCommunity/manim).

**Key Features:**

*   **Precise Animations:** Craft animations with pixel-perfect accuracy using Python code.
*   **Programmatic Control:** Design animations through code, allowing for complex and customizable visuals.
*   **Mathematical Focus:**  Designed specifically for visualizing mathematical concepts.
*   **Community-Driven:** Benefit from an active community, improved features, and enhanced documentation.
*   **Versatile Output:** Create videos suitable for various platforms and educational purposes.
*   **Jupyter Notebook Integration:** Leverage the convenience of `%%manim` magic within JupyterLab notebooks.
*   **Docker Support:** Easily set up and run Manim using Docker containers.

## Table of Contents

*   [Installation](#installation)
*   [Usage](#usage)
*   [Command Line Arguments](#command-line-arguments)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help and Community](#help-and-community)
*   [Contributing](#contributing)
*   [How to Cite](#how-to-cite)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

## Installation

> [!CAUTION]
> These instructions are for the community version _only_. Ensure you are installing the correct version.

For local installation, please visit the [Documentation](https://docs.manim.community/en/stable/installation.html) and follow the appropriate instructions for your operating system.  If you want to try it out first before installing it locally, you can do so [in our online Jupyter environment](https://try.manim.community/).

## Usage

Manim is incredibly versatile. Here's a simple example:

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

Save this code in a file (e.g., `example.py`) and run:

```bash
manim -p -ql example.py SquareToCircle
```

This will render a video showing a square transforming into a circle. Explore more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

## Command Line Arguments

Manim is typically used with the following command structure:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the video, and `-ql` renders at a lower quality.

Other useful flags include:

*   `-s`:  Show the final frame.
*   `-n <number>`: Skip to the *n*th animation.
*   `-f`: Show the file in the file browser.

For a complete list, consult the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

A Docker image (`manimcommunity/manim`) is maintained [on DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Find installation and usage instructions in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help and Community

Get help with installation or usage on our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs or suggest features by opening an issue.

## Contributing

Contributions are welcome, especially for tests and documentation.  See the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines.  Join our [Discord server](https://www.manim.community/discord/) for the latest developments.

Most developers use `uv` for dependency management. Learn more at the [uv documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html).

## How to Cite

To support research, cite Manim in your work. Generate a citation in your preferred format by clicking the "cite this repository" button on the [GitHub repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Read our code of conduct and enforcement details on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and Manim Community Developers (see LICENSE.community).
```
Key improvements and SEO optimizations:

*   **Clear and Concise Hook:** The one-sentence hook provides immediate value.
*   **Keyword Optimization:** Added relevant keywords like "math animations," "visualization," "Python," "animation engine," and "mathematical concepts" throughout the document.
*   **Headings:**  Organized with clear headings for readability and SEO.
*   **Bulleted Key Features:** Easy-to-scan list of benefits.
*   **Internal Linking:**  Links to relevant sections within the README, enhancing user navigation.
*   **External Linking:**  Maintained all original links and formatted them clearly.  Added links to the GitHub repository and other resources for better navigation and SEO.
*   **Emphasis on Community:** Highlights the community aspect.
*   **Call to Action:** Encourages users to try Manim.
*   **Concise Language:**  The content is more focused and efficient.
*   **Alt Text:** Added alt text to images for accessibility and SEO.
*   **Removed Redundancy:** Eliminated unnecessary phrases to make the document more direct.
*   **Markdown Formatting:** Consistent and well-formatted Markdown.
*   **Cite this repository Button:** Added information on how to cite the repo.