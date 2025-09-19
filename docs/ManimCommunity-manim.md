# Manim: Create Stunning Math Animations with Code

**Bring your mathematical explanations to life with Manim, a powerful Python library for generating high-quality animations.** Explore the original repository: [Manim GitHub](https://github.com/ManimCommunity/manim).

<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png"></a>
</p>

Manim empowers you to create dynamic and visually engaging videos for educational content, presentations, and more. This community edition, actively developed and maintained, builds upon the foundation laid by 3Blue1Brown, offering enhanced features and a vibrant community.

**Key Features:**

*   **Programmatic Animation:** Craft animations using Python code for precise control and flexibility.
*   **Versatile Scene Construction:** Build complex scenes with various geometric shapes, mathematical symbols, and textual elements.
*   **Extensive Documentation:** Access comprehensive documentation to guide you through installation, usage, and advanced features.
*   **Community Support:** Join a thriving community on Discord and Reddit to get help, share your creations, and collaborate with other users.
*   **Docker Integration:** Easily set up and run Manim using Docker for simplified installation and environment management.
*   **Jupyter Notebook Support:** Utilize the %%manim magic command for seamless integration and animation within JupyterLab and classic Jupyter notebooks.

## Table of Contents:

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

Manim requires several dependencies; to install, consult the [official documentation](https://docs.manim.community/en/stable/installation.html) for instructions tailored to your operating system. If you'd like to test Manim before installing, try out the [online Jupyter environment](https://try.manim.community/).

## Usage

Create stunning animations by importing `manim` and defining scenes:

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

Save the code in a file, e.g., `example.py`, and execute this command in your terminal:

```sh
manim -p -ql example.py SquareToCircle
```

This will generate and display a video transforming a square into a circle. Explore example scenes in the [GitHub repository](example_scenes) and discover more advanced animations in the [official gallery](https://docs.manim.community/en/stable/examples.html).

Manim also integrates with Jupyter notebooks using the `%%manim` magic command; explore its functionality in the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and experiment with it [online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

Utilize command-line arguments to customize your animations:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

*   `-p`: Preview the video immediately after rendering.
*   `-ql`: Render at a lower quality for faster processing.
*   `-s`: Show the final frame only.
*   `-n <number>`: Skip to the nth animation in a scene.
*   `-f`: Open the file in the file browser.

For a comprehensive list of arguments, consult the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Find detailed documentation at [ReadTheDocs](https://docs.manim.community/) to learn all about Manim.

## Docker

The project offers a Docker image (`manimcommunity/manim`) on [DockerHub](https://hub.docker.com/r/manimcommunity/manim). Learn to install and use it via the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

If you need any help, visit the [Discord Server](https://www.manim.community/discord/) or the [Reddit Community](https://www.reddit.com/r/manim/). Open an issue for bug reports or feature requests.

## Contributing

Contributions are welcome. For contribution guidelines, see the [documentation](https://docs.manim.community/en/stable/contributing.html). Be aware that the project is currently undergoing refactoring, and new feature implementations may not be accepted.

Most developers use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

To properly cite Manim, visit the [repository page](https://github.com/ManimCommunity/manim) and use the "cite this repository" button.

## Code of Conduct

Review the Code of Conduct and its enforcement on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is dual-licensed under the MIT license, with copyrights held by 3blue1brown LLC (LICENSE) and the Manim Community Developers (LICENSE.community).