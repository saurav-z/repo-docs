<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png"></a>
</p>

# Manim: Create Stunning Math Animations with Code

**Bring your mathematical ideas to life with Manim, a powerful and versatile animation engine perfect for creating engaging and explanatory math videos.**

Manim (formerly 3Blue1Brown's Manim) is a Python library that empowers you to generate precise and visually appealing animations programmatically.  This community-maintained version provides enhanced features, improved documentation, and a vibrant community.  Visit the [original repository](https://github.com/ManimCommunity/manim) for more details.

**Key Features:**

*   **Precise Animations:** Create animations with fine-grained control over every element.
*   **Programmatic Control:** Build animations using Python code for flexibility and reproducibility.
*   **Mathematical Visualization:** Easily visualize mathematical concepts, equations, and data.
*   **Community-Driven:** Benefit from an active and supportive community, ongoing development, and comprehensive documentation.
*   **Cross-Platform:** Works on multiple platforms including Windows, Linux, and macOS.

**Key Benefits:**

*   **Ideal for Educators:** Create engaging math lessons and tutorials that captivate your audience.
*   **Perfect for Researchers:** Illustrate complex mathematical concepts and present your findings in an accessible way.
*   **Great for Content Creators:** Produce high-quality videos for YouTube, presentations, and educational materials.

**Key Elements**

*   [Installation](#installation)
*   [Usage](#usage)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help with Manim](#help-with-manim)
*   [Contributing](#contributing)
*   [License](#license)

## Installation

Manim requires dependencies that need to be installed before use. For a quick start, try it out [in our online Jupyter environment](https://try.manim.community/).

For local installation, please visit the [Documentation](https://docs.manim.community/en/stable/installation.html)
and follow the appropriate instructions for your operating system.

## Usage

Manim offers versatility when it comes to animation. Here is an example of a `Scene` to construct:

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

Save the code in a file called `example.py`. Then, run the following in a terminal window:

```sh
manim -p -ql example.py SquareToCircle
```

You should see a video pop up of a square turning into a circle. You can find more examples in the
[GitHub repository](example_scenes). Visit the [official gallery](https://docs.manim.community/en/stable/examples.html) for more advanced examples.

Manim also has a `%%manim` IPython magic, which allows the code to be used in JupyterLab (and Jupyter) notebooks. See the
[corresponding documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and
[try it online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command line arguments

The general usage of Manim is as follows:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag in the command above is for previewing, meaning the video file will automatically open when it is done rendering. The `-ql` flag is for a faster rendering at a lower quality.

Some other useful flags include:

-   `-s` to skip to the end and just show the final frame.
-   `-n <number>` to skip ahead to the `n`'th animation of a scene.
-   `-f` show the file in the file browser.

For a thorough list of command line arguments, visit the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Find complete documentation at [ReadTheDocs](https://docs.manim.community/).

## Docker

The community also maintains a docker image (`manimcommunity/manim`), which can be found [on DockerHub](https://hub.docker.com/r/manimcommunity/manim).
Instructions on how to install and use it can be found in our [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

If you need help installing or using Manim, reach out to our [Discord
Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim). If you want to submit a bug report or feature request, open an issue.

## Contributing

Contributions to Manim are always welcome. Specifically, there is a dire need for tests and documentation. For contribution guidelines, see the [documentation](https://docs.manim.community/en/stable/contributing.html).

However, please note that Manim is undergoing a major refactor. In general,
contributions implementing new features will not be accepted in this period.
The contribution guide may become outdated quickly; we highly recommend joining our
[Discord server](https://www.manim.community/discord/) to discuss any potential
contributions and keep up to date with the latest developments.

Most developers on the project use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

Show your support for the value of Manim by citing it in your research. The best way to cite it is to go to our
[repository page](https://github.com/ManimCommunity/manim) and
click the "cite this repository" button on the right sidebar. This will generate
a citation in your preferred format, and will also integrate well with citation managers.

## Code of Conduct

Read our code of conduct, and how it is enforced on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

The software is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).