<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

# Manim: An Animation Engine for Explanatory Math Videos

**Create stunning mathematical animations and visualizations with Manim, the powerful Python-based animation engine used by 3Blue1Brown and now available through the Manim Community!** ([Original Repo](https://github.com/ManimCommunity/manim))

<p align="center">
  <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
  <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
  <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
  <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
  <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit"></a>
  <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter"></a>
  <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"> </a>
  <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
</p>

## Key Features

*   **Python-Based:**  Write animations using Python, leveraging the language's power and flexibility.
*   **Precise Animations:** Create mathematically accurate and visually appealing animations.
*   **Community-Driven:** Benefit from a vibrant and active community that provides support, resources, and continuous development.
*   **Versatile:**  Suitable for a wide range of applications, from educational videos to scientific visualizations.
*   **Docker Support:** Easily set up and run Manim using Docker containers.
*   **Jupyter Integration:** Seamlessly integrate Manim into Jupyter notebooks with the `%%manim` magic.
*   **Extensive Documentation:** Comprehensive documentation to guide you through every step of the process.

## Getting Started

### Installation

For detailed installation instructions, please consult the [official documentation](https://docs.manim.community/en/stable/installation.html). Choose the installation method appropriate for your operating system.

### Usage

Here's a simple example to get you started:

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

Save the code in a file (e.g., `example.py`) and run it from your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will render and preview your first animation.

### Command Line Arguments

Manim provides a rich set of command-line arguments to control the rendering process:

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality for faster results.
*   `-s`: Show the final frame only.
*   `-n <number>`: Skip to the *n*th animation of a scene.
*   `-f`: Open the file in the file browser.

For a complete list, refer to the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Resources

*   **Documentation:** [ReadTheDocs](https://docs.manim.community/)
*   **Docker:** [DockerHub](https://hub.docker.com/r/manimcommunity/manim)
*   **Jupyter Examples:** [Binder](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
*   **Discord:** [Discord Server](https://www.manim.community/discord/)
*   **Reddit:** [r/manim](https://www.reddit.com/r/manim/)

## Contributing

We welcome contributions! Please consult the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html). Join our [Discord server](https://www.manim.community/discord/) to discuss contributions and stay up-to-date.

## How to Cite

To cite Manim in your work, please use the "cite this repository" button on the [GitHub repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Please review our [Code of Conduct](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).