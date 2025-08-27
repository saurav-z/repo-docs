<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
</p>

# Manim: Create Stunning Explanatory Math Videos

**Bring your mathematical concepts to life with Manim, a powerful Python animation engine!**  This community-driven project allows you to generate precise, visually engaging animations for educational videos, presentations, and more, as famously demonstrated by 3Blue1Brown.  For the original repo, see:  [Manim on GitHub](https://github.com/ManimCommunity/manim).

[![PyPI Latest Release](https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi)](https://pypi.org/project/manim/)
[![Docker Image](https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker)](https://hub.docker.com/r/manimcommunity/manim)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
[![MIT License](https://img.shields.io/badge/license-MIT-red.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit)](https://www.reddit.com/r/manim/)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community)](https://twitter.com/manim_community/)
[![Discord](https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord)](https://www.manim.community/discord/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/manimce/badge/?version=latest)](https://docs.manim.community/)
[![Downloads](https://pepy.tech/badge/manim/month?)](https://pepy.tech/project/manim)
[![CI](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)](https://github.com/ManimCommunity/manim/actions?query=workflow%3ACI)

<hr />

## Key Features

*   **Programmatic Animation:** Create animations using Python code, providing complete control over every aspect of your visuals.
*   **Precise & Customizable:** Build animations with mathematical accuracy, tailored to your exact needs.
*   **Community-Driven:** Benefit from an active and supportive community, with ongoing development and improvements.
*   **Interactive Examples:** Get started quickly with numerous examples and tutorials, including interactive Jupyter Notebooks.
*   **Docker Support:** Easily set up and run Manim using Docker containers.

## Core Features:

*   **Geometric Transformations:** Rotate, scale, translate, and transform objects with ease.
*   **Mathematical Notation:**  Integrate LaTeX and other mathematical notation seamlessly.
*   **Scene Organization:**  Structure your animations with scenes, animations, and objects for a clear workflow.
*   **Flexible Output:** Export your animations as videos, images, or other formats.

## Getting Started

### Installation

To install Manim, please visit the [Installation Guide](https://docs.manim.community/en/stable/installation.html) in the official documentation for detailed instructions, tailored to your operating system. You can also try it out in a [Jupyter Environment](https://try.manim.community/) without installation.

### Example Usage

Here's a simple example of how to create a scene in Manim:

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

This will render and preview a scene transforming a square into a circle.

## Resources

*   **Documentation:** [Manim Documentation](https://docs.manim.community/)
*   **Examples:** [Official Gallery](https://docs.manim.community/en/stable/examples.html)
*   **Jupyter Examples:** [Jupyter Examples](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
*   **Command Line Arguments:** Explore various command-line options for rendering and customization via the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Additional Resources

### Docker

Manim offers a convenient Docker image for easy setup and use.  Find the image on [DockerHub](https://hub.docker.com/r/manimcommunity/manim) and find instructions on how to install and use it at the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

### Help & Community

*   **Discord:**  Connect with other Manim users and get help on the [Discord Server](https://www.manim.community/discord/).
*   **Reddit:**  Join the [Reddit Community](https://www.reddit.com/r/manim/) to ask questions and share your creations.
*   **Issues:**  Submit bug reports and feature requests via [Issues](https://github.com/ManimCommunity/manim/issues).

### Contributing

Contributions to Manim are highly encouraged! See the [Contribution Guidelines](https://docs.manim.community/en/stable/contributing.html) for details.  Note: new feature implementations may be limited during the current refactor.  Join the Discord server for more information and project updates.

#### How to Cite

To properly cite Manim in your work, please visit the [repository page](https://github.com/ManimCommunity/manim) and use the "cite this repository" button on the right sidebar.  This will generate a citation in your preferred format.

### Code of Conduct

Review our [Code of Conduct](https://docs.manim.community/en/stable/conduct.html) to ensure a welcoming and respectful environment for all contributors and users.

### License

Manim is licensed under both the MIT License, with copyright by 3blue1brown LLC (see LICENSE) and Manim Community Developers (see LICENSE.community).