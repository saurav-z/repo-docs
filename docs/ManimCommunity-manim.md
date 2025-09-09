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

# Manim: Create Stunning Math Videos with Python

Manim is a powerful Python library that allows you to programmatically generate beautiful, explanatory math videos, inspired by the work of [3Blue1Brown](https://www.3blue1brown.com/).

**[View the original repository on GitHub](https://github.com/ManimCommunity/manim)**

## Key Features

*   **Precise Animations:** Create animations with pixel-perfect accuracy using mathematical concepts.
*   **Programmatic Control:** Define animations using Python code for maximum flexibility and customization.
*   **Versatile Scene Design:** Build complex scenes with shapes, text, and mathematical expressions.
*   **Community-Driven Development:** Benefit from ongoing improvements, enhanced documentation, and active community support.
*   **Docker Support:** Easily set up and use Manim with Docker.
*   **Jupyter Integration:** Utilize `%%manim` magic for interactive scene creation within Jupyter notebooks.

## Installation

For the community version of Manim, please consult the [official documentation](https://docs.manim.community/en/stable/installation.html) for detailed instructions on installing dependencies and setting up your environment. You can also test it out in our online Jupyter environment [here](https://try.manim.community/).

## Usage

Here's a quick example of how to create a scene:

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

Save the code in a file (e.g., `example.py`) and run it in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

The `-p` flag previews the video, and `-ql` renders it in lower quality.

Explore further examples in the [GitHub repository](example_scenes) or the [official gallery](https://docs.manim.community/en/stable/examples.html).

## Command-Line Arguments

Manim uses command-line arguments to control rendering and output. Here's a useful overview:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

For a full list, refer to the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Manim offers a Docker image (`manimcommunity/manim`) on [DockerHub](https://hub.docker.com/r/manimcommunity/manim). Find instructions on how to install and use it in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help & Community

Get help with Manim and connect with other users via our [Discord Server](https://www.manim.community/discord/) and [Reddit Community](https://www.reddit.com/r/manim/). Submit bug reports or feature requests by opening an issue.

## Contributing

Contributions are welcome! Please see the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines.

## How to Cite

To cite Manim in your work, please use the "cite this repository" button on the [GitHub repository page](https://github.com/ManimCommunity/manim) for a generated citation in your preferred format.

## Code of Conduct

Our Code of Conduct can be found [here](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and Manim Community Developers (see LICENSE.community).