# Manim: Create Stunning Math Animations with Python

**Bring your mathematical concepts to life with Manim, a powerful Python animation engine perfect for creating engaging and explanatory videos!** ([See the original repository](https://github.com/ManimCommunity/manim))

<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

Manim, originally created by Grant Sanderson of 3Blue1Brown, empowers you to generate precise, visually captivating animations programmatically. This community-driven version, ManimCE, offers enhanced features, improved documentation, and a vibrant community for continuous development.

**Key Features:**

*   **Precise Programmatic Control:** Create animations using Python code for ultimate control and flexibility.
*   **Versatile Animation Library:** An extensive library of built-in animations and visual elements.
*   **Ideal for Explanatory Videos:** Perfect for creating engaging math, science, and educational content.
*   **Community-Driven Development:** Benefit from ongoing improvements, active community support, and comprehensive documentation.
*   **Jupyter Notebook Integration:** Seamlessly integrate Manim into your JupyterLab and Jupyter notebooks.
*   **Docker Support:** Easily set up and run Manim with Docker containers.

**What can you do with Manim?**

*   Visualize complex mathematical concepts.
*   Create dynamic diagrams and illustrations.
*   Produce high-quality educational videos.
*   Experiment with animation techniques.

## Table of Contents:

*   [Installation](#installation)
*   [Usage](#usage)
*   [Command Line Arguments](#command-line-arguments)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help & Community](#help-with-manim)
*   [Contributing](#contributing)
*   [How to Cite Manim](#how-to-cite-manim)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

## Installation

To get started, install Manim and its dependencies following the instructions in the official [Documentation](https://docs.manim.community/en/stable/installation.html).

If you'd like to experiment without installing locally, you can use our online [Jupyter environment](https://try.manim.community/).

## Usage

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

Save the code in a file (e.g., `example.py`) and then run the following command in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will generate a video showing a square morphing into a circle.

Find more examples and explore advanced features in the [official gallery](https://docs.manim.community/en/stable/examples.html) and the [GitHub repository](example_scenes).

**Jupyter Notebooks:** Take advantage of the `%%manim` IPython magic to use Manim in JupyterLab and classic Jupyter notebooks.

## Command Line Arguments

Control the rendering process using command-line arguments:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

*   `-p`: Preview the video automatically.
*   `-ql`: Render at a lower quality for faster processing.
*   `-s`: Show the final frame.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Show the file in the file browser.

For detailed information on available arguments, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Use the community-maintained Docker image (`manimcommunity/manim`) to easily run Manim. Find it on [DockerHub](https://hub.docker.com/r/manimcommunity/manim), and consult the [documentation](https://docs.manim.community/en/stable/installation/docker.html) for installation and usage instructions.

## Help & Community

Get help, discuss Manim, and stay connected with the community on our [Discord Server](https://www.manim.community/discord/) and [Reddit Community](https://www.reddit.com/r/manim/). Report bugs or request features by opening an issue on the [GitHub repository](https://github.com/ManimCommunity/manim).

## Contributing

Contributions are welcome! Please review the [documentation](https://docs.manim.community/en/stable/contributing.html) for contribution guidelines.
Join the [Discord server](https://www.manim.community/discord/) to discuss potential contributions.

Most developers use `uv` for dependency management; see the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation for setup instructions.

## How to Cite Manim

Support the project by citing Manim in your work. Find citation instructions on the [repository page](https://github.com/ManimCommunity/manim) under the "Cite this repository" button.

## Code of Conduct

Read the full code of conduct and enforcement details on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

The software is dual-licensed under the MIT license and Manim Community Developers (see LICENSE.community).