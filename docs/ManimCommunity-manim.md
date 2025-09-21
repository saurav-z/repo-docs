<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

<h1 align="center">Manim: An Open-Source Animation Engine for Explanatory Math Videos</h1>

**Bring your mathematical ideas to life with Manim, a powerful Python-based animation engine perfect for creating stunning visuals for educational content.** This is the community edition of Manim, actively maintained with enhanced features and a vibrant community.  Explore the original project at [3b1b/manim](https://github.com/3b1b/manim).

<br/>
<p align="center">
    <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
    <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit" href=></a>
    <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter">
    <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"> </a>
    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
</p>

<hr />

## Key Features

*   **Create Precise Animations:**  Programmatically generate complex mathematical animations.
*   **Versatile for Education:** Ideal for creating engaging visuals for math, physics, and other technical subjects.
*   **Python-Based:**  Leverage the power and flexibility of the Python programming language.
*   **Community-Driven:** Benefit from an active community, continuous development, and extensive documentation.
*   **Cross-Platform:**  Runs on various operating systems.
*   **Jupyter Notebook Integration:** Seamlessly integrate Manim into your JupyterLab and Jupyter notebooks using the `%%manim` magic.
*   **Docker Support:** Easily set up and run Manim using Docker containers.

## Getting Started

### Installation

For detailed installation instructions, please consult the official [Manim Documentation](https://docs.manim.community/en/stable/installation.html).  You can also try Manim directly in your browser using the [online Jupyter environment](https://try.manim.community/).

### Basic Usage

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

Save this code as `example.py` and run it from your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This command will render and preview a simple animation transforming a square into a circle.

### Command-Line Arguments

Manim's command-line interface offers several options:

*   `-p`: Preview the output automatically.
*   `-ql`: Render at a lower quality for faster processing.
*   `-s`: Show the final frame only.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Open the output file in your file browser.

For a comprehensive list, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Further Resources

*   **Documentation:** Explore the comprehensive documentation at [ReadTheDocs](https://docs.manim.community/).
*   **Examples:** Discover advanced examples in the [official gallery](https://docs.manim.community/en/stable/examples.html).
*   **Docker:** Learn to use the official [Docker image](https://hub.docker.com/r/manimcommunity/manim) for easy setup.
*   **Jupyter Examples:**  Get started with JupyterLab notebooks using the example scenes in the  [GitHub repository](https://github.com/ManimCommunity/manim).

## Community and Support

*   **Discord:** Join the community on the [Discord Server](https://www.manim.community/discord/) for help and discussions.
*   **Reddit:** Connect with other users on the [Reddit Community](https://www.reddit.com/r/manim/).
*   **Issues:** Submit bug reports or feature requests on [GitHub Issues](https://github.com/ManimCommunity/manim/issues).

## Contributing

We welcome contributions! Please see the [contributing guidelines](https://docs.manim.community/en/stable/contributing.html) for details. Note that Manim is currently undergoing a major refactor, so contributions focusing on new features may not be accepted during this period.  Join the [Discord server](https://www.manim.community/discord/) for the latest updates.

## How to Cite

To properly acknowledge the use of Manim in your work, please cite the project using the citation information available on the [GitHub repository page](https://github.com/ManimCommunity/manim) by clicking the "Cite this repository" button on the right sidebar.

## Code of Conduct

Please review our Code of Conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

This software is dual-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and copyright by Manim Community Developers (see LICENSE.community).