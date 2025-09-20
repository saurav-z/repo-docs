<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
    <br />
</p>

# Manim: Animate Your Math!

**Create stunning, explanatory math videos with Manim, the powerful animation engine used by 3Blue1Brown.**  This community-driven version of Manim provides a robust and accessible platform for generating high-quality mathematical visualizations and educational content. Learn more and contribute at the [Manim Community GitHub Repository](https://github.com/ManimCommunity/manim).

[![PyPI Latest Release](https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi)](https://pypi.org/project/manim/)
[![Docker Image](https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker)](https://hub.docker.com/r/manimcommunity/manim)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
[![MIT License](https://img.shields.io/badge/license-MIT-red.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit)](https://www.reddit.com/r/manim/)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community)](https://twitter.com/manim_community/)
[![Discord](https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord)](https://www.manim.community/discord/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/manimce/badge/?version=latest)](https://docs.manim.community/)
[![Downloads](https://pepy.tech/badge/manim/month?)](https://pepy.tech/project/manim)
[![CI](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)](https://github.com/ManimCommunity/manim/workflows/CI)

---

Manim is an animation engine perfect for creating compelling explanatory math videos, inspired by the visual style of 3Blue1Brown.  This community edition (ManimCE) offers ongoing development, a supportive community, and extensive documentation.

**Key Features:**

*   **Programmatic Animation:** Create precise animations using Python code, giving you complete control over every aspect of your visuals.
*   **Mathematical Visualization:** Easily represent and animate mathematical concepts, equations, and diagrams.
*   **Community Driven:** Benefit from active development, improvements, and community support.
*   **Cross-Platform Compatibility:** Available for use on multiple operating systems.
*   **Jupyter Integration:** Utilize the `%%manim` IPython magic to seamlessly integrate Manim animations into your Jupyter notebooks.
*   **Docker Support:** Easily set up and run Manim with pre-configured Docker images.

## Table of Contents:

*   [Installation](#installation)
*   [Usage](#usage)
*   [Command Line Arguments](#command-line-arguments)
*   [Documentation](#documentation)
*   [Docker](#docker)
*   [Help with Manim](#help-with-manim)
*   [Contributing](#contributing)
*   [How to Cite Manim](#how-to-cite-manim)
*   [Code of Conduct](#code-of-conduct)
*   [License](#license)

## Installation

> [!CAUTION]
> These instructions are for the community version _only_. Trying to use these instructions to install [3b1b/manim](https://github.com/3b1b/manim) or instructions there to install this version will cause problems. Read [this](https://docs.manim.community/en/stable/faq/installation.html#why-are-there-different-versions-of-manim) and decide which version you wish to install, then only follow the instructions for your desired version.

Manim requires several dependencies.  To get started quickly, try it out online using our [Jupyter environment](https://try.manim.community/).

For local installation, consult the detailed instructions in the [official Documentation](https://docs.manim.community/en/stable/installation.html) for your specific operating system.

## Usage

Manim is designed for versatility. Here's a simple example `Scene`:

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

Save the code to `example.py`, then run in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will render a video showing a square morphing into a circle. Check out the [example_scenes](example_scenes) or the [official gallery](https://docs.manim.community/en/stable/examples.html) for more complex examples.

You can also use `%%manim` in JupyterLab/Jupyter notebooks.  See the [IPython magic documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and try it out [online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command Line Arguments

The general usage of Manim involves the command line tool:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

The `-p` flag previews the video. `-ql` renders at low quality for speed.

Useful flags:

*   `-s`:  Show the final frame.
*   `-n <number>`: Skip to the `n`'th animation.
*   `-f`: Show the file in the file browser.

Find a comprehensive list of arguments in the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Use the official Docker image (`manimcommunity/manim`) from [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Instructions are available in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Help with Manim

Get support and interact with the community on our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim).  Report bugs or suggest features by opening an issue.

## Contributing

Contributions are welcome!  We especially need help with tests and documentation. See the [contributing documentation](https://docs.manim.community/en/stable/contributing.html) for details.

Please note that Manim is currently undergoing a major refactor, and contributions that implement new features may not be accepted at this time. Join the [Discord server](https://www.manim.community/discord/) for the latest developments.

Most developers use `uv` for dependency management.  Install it and familiarize yourself with it.  See the [uv documentation](https://docs.astral.sh/uv/) and the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

We encourage the use of Manim in research and value effective communication. To cite Manim, go to our [repository page](https://github.com/ManimCommunity/manim) and click the "cite this repository" button on the right sidebar. This generates a citation in your preferred format.

## Code of Conduct

Our Code of Conduct is available on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is dual-licensed under the MIT license (copyright 3blue1brown LLC and the Manim Community Developers).
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:**  A strong, one-sentence description that immediately grabs the user's attention.
*   **Keywords:**  Uses relevant keywords like "animation engine," "math videos," "mathematical visualization," and "Python" to improve search ranking.
*   **Structured Headings and Subheadings:** Improves readability and allows for easy navigation, improving SEO.
*   **Bulleted Key Features:** Highlights the most important aspects of the software, quickly conveying its value.
*   **Contextual Links:**  Includes links to the official website, documentation, and community resources.  Links are optimized.
*   **Alt Text:** Included `alt` text for all images to improve accessibility and SEO.
*   **Community Focus:**  Highlights the community aspect of the project, which is a key differentiator.
*   **Call to Action:** Encourages users to contribute and provides resources.
*   **Citation Information:**  Highlights how to properly cite the project.
*   **Readability:** Improved formatting and writing to make the README easier to understand.
*   **Conciseness:** The README has been trimmed down to only include critical information.
*   **Markdown Formatting:** Uses markdown for consistent styling.