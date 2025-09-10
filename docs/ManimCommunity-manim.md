<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
    <br />
    <br />
</p>

# Manim: Create Stunning Explanatory Math Videos

**Transform complex mathematical concepts into captivating animations with Manim, an open-source animation engine.**

Manim allows you to create precise, visually appealing animations for educational and explanatory videos, similar to those made by [3Blue1Brown](https://www.3blue1brown.com/). This community-driven version (ManimCE) offers ongoing development, improved features, and a vibrant community.

**[See the original Manim repository](https://github.com/ManimCommunity/manim)**

**Key Features:**

*   **Programmatic Animation:** Create animations using Python code for complete control and precision.
*   **Mathematical Visualization:** Easily represent and animate mathematical objects, equations, and concepts.
*   **Community-Driven:** Benefit from active development, enhanced features, and a supportive community.
*   **Versatile:** From basic shapes to complex simulations, Manim handles a wide range of animation needs.
*   **Cross-Platform:** Run Manim on Windows, macOS, and Linux.
*   **Integration:** Use Manim within Jupyter notebooks with the `%%manim` magic command.
*   **Docker Support:** Easily set up and run Manim using pre-built Docker images.

**Quick Links:**

*   [Documentation](https://docs.manim.community/)
*   [Examples](https://docs.manim.community/en/stable/examples.html)
*   [Discord Server](https://www.manim.community/discord/)
*   [Reddit Community](https://www.reddit.com/r/manim/)
*   [DockerHub](https://hub.docker.com/r/manimcommunity/manim)
*   [Jupyter Examples](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
*   [MIT License](http://choosealicense.com/licenses/mit/)
*   [PyPI](https://pypi.org/project/manim/)

**Table of Contents:**

-   [Installation](#installation)
-   [Usage](#usage)
-   [Command Line Arguments](#command-line-arguments)
-   [Docker](#docker)
-   [Help & Community](#help-with-manim)
-   [Contributing](#contributing)
-   [How to Cite Manim](#how-to-cite-manim)
-   [Code of Conduct](#code-of-conduct)
-   [License](#license)

## Installation

To install Manim, please visit the [Documentation](https://docs.manim.community/en/stable/installation.html) and follow the instructions for your operating system. If you want to try it out without local installation, you can do so [in our online Jupyter environment](https://try.manim.community/).

## Usage

Here's a basic example to get you started:

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

Save this code as `example.py` and run:

```bash
manim -p -ql example.py SquareToCircle
```

This will render and display a video transforming a square into a circle.  Explore more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

## Command Line Arguments

Manim's general usage is as follows:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the video.  `-ql` renders at a lower quality for faster rendering.

Useful flags:

*   `-s`:  Show only the final frame.
*   `-n <number>`: Skip to the *n*th animation.
*   `-f`: Open the file in the file browser.

For a comprehensive list, consult the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Docker

A Docker image is available on [DockerHub](https://hub.docker.com/r/manimcommunity/manim). Find instructions in our [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help & Community

For help, join the [Discord Server](https://www.manim.community/discord/) or the [Reddit Community](https://www.reddit.com/r/manim/). Report bugs and suggest features by opening an issue.

## Contributing

Contributions are welcome! See the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines. Note that significant refactoring is underway, and new feature contributions may be limited.  Join the [Discord server](https://www.manim.community/discord/) to discuss contributions and stay updated.

## How to Cite Manim

To cite Manim, visit the [repository page](https://github.com/ManimCommunity/manim) and click the "cite this repository" button for a citation in your preferred format.

## Code of Conduct

Read our full code of conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and by Manim Community Developers (see LICENSE.community).