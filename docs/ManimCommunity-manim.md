<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
    <br />
    <br />
    <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
    <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit" href="#"></a>
    <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter">
    <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"> </a>
    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
    <br />
    <br />
</p>

# Manim: Create Stunning Explanatory Math Videos with Python

Manim is a powerful Python library for generating mathematical animations, perfect for creating engaging educational videos. ([View the source on GitHub](https://github.com/ManimCommunity/manim))

**Key Features:**

*   **Programmatic Animation:** Create precise animations using Python code.
*   **Versatile Scene Creation:** Build complex scenes with various geometric shapes, mathematical equations, and text.
*   **Customizable Animations:** Control every aspect of your animations, from movement and transformations to color and style.
*   **Community-Driven:** Actively maintained and developed by a vibrant community, ensuring continuous improvements and support.
*   **Jupyter Integration:** Seamlessly integrate Manim into Jupyter notebooks for interactive animation development.
*   **Docker Support:** Easily set up and use Manim with Docker containers.

## Table of Contents

-   [Installation](#installation)
-   [Usage](#usage)
-   [Command Line Arguments](#command-line-arguments)
-   [Documentation](#documentation)
-   [Docker](#docker)
-   [Help with Manim](#help-with-manim)
-   [Contributing](#contributing)
-   [How to Cite Manim](#how-to-cite-manim)
-   [Code of Conduct](#code-of-conduct)
-   [License](#license)

## Installation

To get started with Manim, you'll need to install it along with its dependencies.  For detailed instructions on installation, please visit the [Manim Documentation](https://docs.manim.community/en/stable/installation.html) and follow the appropriate instructions for your operating system.  If you want to try Manim before installing it locally, you can do so [in our online Jupyter environment](https://try.manim.community/).

## Usage

Here's a simple example to illustrate how to use Manim:

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

Save this code in a file named `example.py` and run the following command in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will render a video showing a square transforming into a circle. For more examples, check out the [example_scenes](https://github.com/ManimCommunity/manim/tree/main/example_scenes) within this GitHub repository or the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also supports the `%%manim` IPython magic for use in JupyterLab and classic Jupyter notebooks. You can learn more in the [IPython magic documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and [try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

Manim's command-line interface provides various options for rendering and customizing your animations.  Here's a basic illustration:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

*   `-p`: Previews the video automatically after rendering.
*   `-ql`: Renders the video at a lower quality for faster processing.
*   `-s`: Shows only the final frame.
*   `-n <number>`: Skips to the *n*th animation in a scene.
*   `-f`: Opens the output file in the file browser.

For a complete list of command-line arguments, see the [configuration guide](https://docs.manim.community/en/stable/guides/configuration.html) in the documentation.

## Documentation

Extensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Manim provides a Docker image for easy setup and use.  The image is available on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  For detailed instructions on how to install and use the Docker image, please consult the [Docker installation guide](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Need help with Manim?  Join the community!  You can find assistance on the [Discord Server](https://www.manim.community/discord/) and the [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs or suggest new features by opening an issue.

## Contributing

Contributions to Manim are welcome!  The project is actively seeking contributions in areas such as testing and documentation.  Refer to the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html) for further details.

However, the project is currently undergoing a major refactor. New features will not be accepted during this time.  We highly recommend joining our [Discord server](https://www.manim.community/discord/) for the latest updates and to discuss potential contributions.

For help with development, the project uses `uv` for package management. More information on `uv` is available [here](https://docs.astral.sh/uv/). For installation with `uv`, check the [dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

Please cite Manim in your work to acknowledge the software's value.  To generate a citation in your preferred format, visit the [repository page](https://github.com/ManimCommunity/manim) and click the "cite this repository" button on the right sidebar.

## Code of Conduct

Adhere to our [Code of Conduct](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is available under the MIT License and the Manim Community Developers License, ensuring open-source use and community contributions. See [LICENSE](https://github.com/ManimCommunity/manim/blob/main/LICENSE) and [LICENSE.community](https://github.com/ManimCommunity/manim/blob/main/LICENSE.community).