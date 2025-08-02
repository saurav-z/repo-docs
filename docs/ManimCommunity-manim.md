<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
    <br />
    <br />
</p>

<p align="center">
    <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
    <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit" href=></a>
    <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter">
    <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"> </a>
    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
    <br />
</p>

# Manim: Create Stunning Explanatory Math Videos

**Manim is a powerful Python library for generating mathematical animations, empowering you to visually explain complex concepts.**

<hr>

## Key Features

*   **Programmatic Animation:** Create animations using Python code for precise control.
*   **Mathematical Visualization:** Visualize equations, graphs, and geometric objects with ease.
*   **Customization:**  Tailor animations to your specific needs with extensive customization options.
*   **Community-Driven:** Benefit from an active community, comprehensive documentation, and regular updates.
*   **Cross-Platform:** Supports various operating systems through Python and Docker.

## What is Manim?

Manim is an animation engine designed for generating high-quality videos to explain mathematical concepts. It's perfect for educators, researchers, and anyone wanting to create visually engaging educational content. Manim is used by creators like [3Blue1Brown](https://www.3blue1brown.com/) for creating explanatory math videos.

### Important Note:

This is the community edition of Manim (ManimCE). It is a maintained and developed fork of the original Manim project. For the original version, see [3b1b/manim](https://github.com/3b1b/manim).

## Getting Started

### Installation

For detailed installation instructions, please refer to the [official documentation](https://docs.manim.community/en/stable/installation.html). Consider using the online Jupyter environment available at [https://try.manim.community/](https://try.manim.community/) for a quick try before installing locally.

### Example Usage

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

Save this code as `example.py` and run the following command in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will render and display a square transforming into a circle! More examples are available in the [GitHub repository](example_scenes) and in the [official gallery](https://docs.manim.community/en/stable/examples.html).

### Jupyter Notebook Integration

Manim seamlessly integrates with JupyterLab (and classic Jupyter) notebooks using the `%%manim` IPython magic. Check out the [documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and try it out [online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

### Command-Line Arguments

Manim uses command-line arguments to control various aspects of rendering.

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/main/docs/source/_static/command.png)

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality (faster).
*   `-s`: Show the final frame only.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Show the file in the file browser.

For a complete list of options, consult the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Resources

*   **Documentation:** Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).
*   **Docker:**  Use the official Docker image: `manimcommunity/manim` ([DockerHub](https://hub.docker.com/r/manimcommunity/manim) and [documentation](https://docs.manim.community/en/stable/installation/docker.html)).
*   **Community:** Get help and connect with other users on our [Discord Server](https://www.manim.community/discord/) and [Reddit Community](https://www.reddit.com/r/manim/).

## Contributing

Contributions are welcome! Refer to the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines.

**Note:** Manim is currently undergoing a major refactor. New feature contributions are generally not accepted during this period. The best way to stay updated is to join the [Discord server](https://www.manim.community/discord/).

## How to Cite Manim

Cite Manim to support research!  Go to the [GitHub repository page](https://github.com/ManimCommunity/manim) and click the "cite this repository" button to generate a citation in your preferred format.

## Code of Conduct

Read our code of conduct [here](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is dual-licensed under the [MIT license](https://github.com/ManimCommunity/manim/blob/main/LICENSE), copyrighted by both 3blue1brown LLC and Manim Community Developers.