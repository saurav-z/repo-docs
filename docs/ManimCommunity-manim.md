<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
</p>

# Manim: Create Stunning Math Animations with Code

**Bring your mathematical ideas to life with Manim, a powerful Python library for generating beautiful and precise animations, perfect for educational videos and visualizations.** ([Original Repository](https://github.com/ManimCommunity/manim))

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

---

## Key Features:

*   **Programmatic Animation:** Control every aspect of your animations with Python code for unparalleled precision.
*   **Versatile Scene Creation:** Build complex scenes with geometric shapes, mathematical expressions, and custom objects.
*   **High-Quality Output:** Render stunning videos and images for educational content, presentations, and research.
*   **Community-Driven Development:** Benefit from an active community, regular updates, and extensive documentation.
*   **Jupyter Notebook Integration:** Utilize the `%%manim` IPython magic for seamless animation creation within Jupyter environments.

## Getting Started

Manim is an extremely versatile package. Here's an example `Scene` you can construct:

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

To view the output, save the code in a file called `example.py` and run in a terminal window:

```sh
manim -p -ql example.py SquareToCircle
```

## Installation

> [!CAUTION]
> These instructions are for the community version _only_.  Read [this](https://docs.manim.community/en/stable/faq/installation.html#why-are-there-different-versions-of-manim) and decide which version you wish to install, then only follow the instructions for your desired version.

For detailed installation instructions, including dependency management, please refer to the [official documentation](https://docs.manim.community/en/stable/installation.html).  You can also try Manim out online in our [Jupyter environment](https://try.manim.community/).

## Usage and Command Line Arguments

Learn the basics of Manim with this code example, and discover how to use command-line arguments to control rendering.
![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)
Some useful flags include:
*   `-p` for previewing (opens video automatically).
*   `-ql` for faster, lower-quality rendering.
*   `-s` to show the final frame.
*   `-n <number>` to skip to a specific animation.
*   `-f` to show the file in the file browser.
  
For a thorough list of command line arguments, visit the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Use the pre-built Docker image for easy setup and deployment:  `manimcommunity/manim` on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Installation and usage instructions can be found in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Community & Support

*   **Discord:** [Discord Server](https://www.manim.community/discord/)
*   **Reddit:** [Reddit Community](https://www.reddit.com/r/manim/)
*   **Issues:**  Submit bug reports and feature requests by opening an issue.

## Contributing

Contributions are welcome! Please review the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html), and join the [Discord server](https://www.manim.community/discord/) to discuss potential contributions.

Currently, contributions implementing new features will not be accepted due to an ongoing refactor.
Most developers on the project use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

To support the project, please cite Manim in your work by using the "cite this repository" button on the [repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Review our [Code of Conduct](https://docs.manim.community/en/stable/conduct.html) for community guidelines.

## License

Manim is double-licensed under the MIT license (3blue1brown LLC) and the MIT license (Manim Community Developers). See [LICENSE](https://github.com/ManimCommunity/manim/blob/main/LICENSE) for details.