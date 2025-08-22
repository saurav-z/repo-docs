<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
    <br />
    <br />
</p>

# Manim: The Animation Engine for Explanatory Math Videos

**Bring your mathematical ideas to life with Manim, a powerful Python-based animation engine.** Create stunning visuals for educational videos, presentations, and more.  ([See the original repo](https://github.com/ManimCommunity/manim))

**Key Features:**

*   **Programmatic Animation:** Create animations with precise control using Python code.
*   **Versatile Scene Creation:** Build complex scenes with geometric objects, text, and mathematical formulas.
*   **Community-Driven Development:** Benefit from a vibrant and active community.
*   **Integration with Jupyter:** Seamlessly use Manim in JupyterLab and classic Jupyter notebooks.
*   **Extensive Documentation:** Comprehensive documentation to guide you through every step.
*   **Docker Support:** Easily set up and run Manim using Docker containers.

---

## Table of Contents

*   [What is Manim?](#manim-the-animation-engine-for-explanatory-math-videos)
*   [Key Features](#key-features)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Command Line Arguments](#command-line-arguments)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help and Community](#help-and-community)
*   [Contributing](#contributing)
*   [How to Cite Manim](#how-to-cite-manim)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

---

## What is Manim?

Manim is an animation engine specifically designed for creating high-quality explanatory math videos, as popularized by the 3Blue1Brown YouTube channel. It allows you to programmatically generate animations, giving you precise control over every element and movement. This community edition (ManimCE) focuses on continued development, improved features, and an active community.

---

## Installation

To get started with Manim, follow the installation instructions on the [Documentation](https://docs.manim.community/en/stable/installation.html) page. Choose the instructions for the *community version* to avoid any compatibility issues. If you'd like to try it out first, you can use the [online Jupyter environment](https://try.manim.community/).

---

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

Save the code in a file named `example.py` and run the following command in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This command will generate and preview a video of a square transforming into a circle. Explore the [official gallery](https://docs.manim.community/en/stable/examples.html) for more advanced examples.  You can also use the `%%manim` IPython magic within Jupyter notebooks.

---

## Command Line Arguments

Manim offers a variety of command-line arguments to customize your animations.  For a basic render, use:

```bash
manim <your_file.py> <SceneName>
```

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality for faster results.
*   `-s`: Show only the final frame.
*   `-n <number>`: Skip to the nth animation in the scene.
*   `-f`: Show the file in the file browser.

Refer to the [documentation](https://docs.manim.community/en/stable/guides/configuration.html) for a comprehensive list of arguments.

---

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

---

## Docker

Manim also provides a Docker image (`manimcommunity/manim`) [on DockerHub](https://hub.docker.com/r/manimcommunity/manim).  See the [documentation](https://docs.manim.community/en/stable/installation/docker.html) for instructions on how to use it.

---

## Help and Community

Get help and connect with other Manim users:

*   [Discord Server](https://www.manim.community/discord/)
*   [Reddit Community](https://www.reddit.com/r/manim/)

For bug reports or feature requests, please open an issue.

---

## Contributing

Contributions are welcome! The project is currently undergoing a major refactor; we recommend joining the [Discord server](https://www.manim.community/discord/) to discuss contributions and stay updated. For guidelines, see the [documentation](https://docs.manim.community/en/stable/contributing.html). Use `uv` for project management. Find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html).

---

## How to Cite Manim

To properly cite Manim in your work, go to the [repository page](https://github.com/ManimCommunity/manim) and click the "cite this repository" button to generate a citation in your preferred format.

---

## Code of Conduct

Read the full Code of Conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

---

## License

Manim is dual-licensed under the MIT license (copyright 3blue1brown LLC and Manim Community Developers). See `LICENSE` and `LICENSE.community` for details.