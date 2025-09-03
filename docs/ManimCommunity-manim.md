<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

# Manim: Create Stunning Math Animations

**Bring your math explanations to life with Manim, an open-source animation engine for creating visually engaging explanatory videos.**

[View the original repository on GitHub](https://github.com/ManimCommunity/manim)

[![PyPI Latest Release](https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi)](https://pypi.org/project/manim/)
[![Docker Image](https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker)](https://hub.docker.com/r/manimcommunity/manim)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
[![MIT License](https://img.shields.io/badge/license-MIT-red.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit)](https://www.reddit.com/r/manim/)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community)](https://twitter.com/manim_community/)
[![Discord](https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord)](https://www.manim.community/discord/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/manimce/badge/?version=latest)](https://docs.manim.community/)
[![Downloads](https://pepy.tech/badge/manim/month?)](https://pepy.tech/project/manim)
[![CI](https://github.com/ManimCommunity/manim/workflows/CI/badge.svg)](https://github.com/ManimCommunity/manim/workflows/CI)

Manim is a powerful Python library for creating animated mathematical visualizations, similar to those used by [3Blue1Brown](https://www.3blue1brown.com/). This community-maintained version (ManimCE) offers enhanced features, an active community, and continuous development.

**Key Features:**

*   **Programmatic Animation:** Create precise animations with code, giving you full control over every detail.
*   **Versatile Scene Creation:** Build complex scenes with a wide range of geometric objects, transformations, and effects.
*   **Cross-Platform Compatibility:** Works on Windows, macOS, and Linux.
*   **Jupyter Notebook Integration:** Utilize `%%manim` magic for seamless integration in Jupyter environments.
*   **Active Community:** Benefit from a supportive community on Discord and Reddit, plus extensive documentation.
*   **Docker Support:** Easily set up your environment with pre-built Docker images.

## Getting Started

### Installation

For detailed installation instructions, please consult the [official documentation](https://docs.manim.community/en/stable/installation.html). Consider using the [online Jupyter environment](https://try.manim.community/) to test Manim before installing.

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

Save the code as `example.py` and run in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

## Command Line Arguments

Use flags to control how your animations are rendered. For example:

*   `-p`: Preview the video.
*   `-ql`: Render at a lower quality (faster).
*   `-s`: Show the final frame.
*   `-n <number>`: Skip to a specific animation.

For a comprehensive list, refer to the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Resources

*   **Documentation:** [Read the Docs](https://docs.manim.community/)
*   **Examples:** Explore the [official gallery](https://docs.manim.community/en/stable/examples.html) and the [GitHub repository](example_scenes) for inspiration.
*   **Docker:** [DockerHub](https://hub.docker.com/r/manimcommunity/manim)
*   **Jupyter Notebook Examples:** [Binder](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)

## Support and Community

*   **Discord:** [Join our Discord server](https://www.manim.community/discord/)
*   **Reddit:** [Join the Reddit Community](https://www.reddit.com/r/manim/)
*   **Issues:** Report bugs or request features by opening an issue on [GitHub](https://github.com/ManimCommunity/manim).

## Contributing

Contributions are welcome! Please see the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html) for details. Join our [Discord server](https://www.manim.community/discord/) to discuss contributions.

Use `uv` for development. Install it and find out how to install Manim with it at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the Manim documentation.

## How to Cite

To properly acknowledge the use of Manim in your work, please use the "cite this repository" button on the [GitHub repository page](https://github.com/ManimCommunity/manim) to generate a citation in your preferred format.

## Code of Conduct

Our code of conduct can be read on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license (copyright by 3blue1brown LLC and Manim Community Developers). See the [LICENSE](https://github.com/ManimCommunity/manim/blob/main/LICENSE) files for details.