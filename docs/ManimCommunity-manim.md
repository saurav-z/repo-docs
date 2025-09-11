<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png"></a>
</p>

# Manim: Create Stunning Explanatory Math Videos

**Bring your mathematical concepts to life with Manim, the powerful and versatile animation engine for creating visually engaging math videos.**  This is the community-maintained version of the original Manim project.  ([View the original repo](https://github.com/ManimCommunity/manim)).

<br/>

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

<hr/>

Manim allows you to programmatically create precise and visually appealing animations, perfect for explaining complex mathematical concepts.  It's used by creators like 3Blue1Brown to produce engaging educational content.

**Key Features:**

*   **Precise Animations:** Control every aspect of your animations using code.
*   **Mathematical Visualization:** Easily represent and manipulate mathematical objects.
*   **Versatile Scene Creation:** Build complex scenes with animations, text, and more.
*   **Community Driven:** Benefit from active development, improved features, and comprehensive documentation.
*   **Jupyter Notebook Integration:** Seamlessly use Manim within Jupyter environments.

## Table of Contents:

-   [Installation](#installation)
-   [Usage](#usage)
-   [Documentation](#documentation)
-   [Docker](#docker)
-   [Help with Manim](#help-with-manim)
-   [Contributing](#contributing)
-   [How to Cite Manim](#how-to-cite-manim)
-   [Code of Conduct](#code-of-conduct)
-   [License](#license)

## Installation

> [!CAUTION]
> These instructions are for the community version _only_. Ensure you're following instructions for this version and not [3b1b/manim](https://github.com/3b1b/manim) to avoid issues.  See the [FAQ](https://docs.manim.community/en/stable/faq/installation.html#why-are-there-different-versions-of-manim) for more details.

Manim requires certain dependencies. You can test Manim online using [our online Jupyter environment](https://try.manim.community/) before local installation.

For local installation, follow the instructions in the [Documentation](https://docs.manim.community/en/stable/installation.html) specific to your operating system.

## Usage

Manim provides great versatility. Here's an example of a `Scene`:

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

Save this code in `example.py` and run the command:

```sh
manim -p -ql example.py SquareToCircle
```

This will generate a video where a square transforms into a circle.  Explore more examples in the [GitHub repository](example_scenes) or the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also offers a `%%manim` IPython magic for convenient use in JupyterLab and classic Jupyter notebooks. See the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and [try it online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

Manim uses the following command structure:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the video, and `-ql` renders a lower-quality, faster version.

Other useful flags:

*   `-s`: Show the final frame.
*   `-n <number>`: Skip to the *n*th animation.
*   `-f`: Open the file in the file browser.

Refer to the [documentation](https://docs.manim.community/en/stable/guides/configuration.html) for a complete list of command-line arguments.

## Documentation

Find comprehensive documentation at [ReadTheDocs](https://docs.manim.community/).

## Docker

A Docker image (`manimcommunity/manim`) is available [on DockerHub](https://hub.docker.com/r/manimcommunity/manim).  See the [documentation](https://docs.manim.community/en/stable/installation/docker.html) for installation and usage instructions.

## Help with Manim

Get help and connect with the community on our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/).  Report bugs and request features by opening an issue.

## Contributing

Contributions are welcome!  We especially need help with tests and documentation. See the [documentation](https://docs.manim.community/en/stable/contributing.html) for contribution guidelines.

Please note that Manim is undergoing a refactor, so contributions implementing new features are generally not accepted. Join our [Discord server](https://www.manim.community/discord/) for the latest updates.

Most developers use `uv` for package management. Learn more at the [uv documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

Cite Manim in your work to demonstrate its value. Use the "cite this repository" button on the [repository page](https://github.com/ManimCommunity/manim) to generate a citation in your preferred format.

## Code of Conduct

Read our full code of conduct and enforcement details on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Double-licensed under the MIT license (copyright by 3blue1brown LLC and Manim Community Developers). See `LICENSE` and `LICENSE.community`.