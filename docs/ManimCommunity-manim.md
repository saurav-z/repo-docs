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
</p>

# Manim: Create Stunning Explanatory Math Videos

**Manim is a powerful animation engine, perfect for creating engaging and informative math and science educational videos.**

[Visit the original repository on GitHub](https://github.com/ManimCommunity/manim)

## Key Features

*   **Programmatic Animation:** Create precise animations using Python code.
*   **Versatile:** Suitable for a wide range of mathematical and scientific concepts.
*   **Community Driven:** Actively maintained and developed by a passionate community.
*   **High-Quality Output:** Produce professional-looking videos for educational or presentation purposes.
*   **Jupyter Notebook Integration:** Seamlessly integrate Manim into your JupyterLab notebooks.
*   **Docker Support:** Easily run Manim in a Docker container.

## Getting Started

### Installation

For detailed installation instructions, please see the [official documentation](https://docs.manim.community/en/stable/installation.html). Choose the appropriate instructions for your operating system.

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

Save the code in a file named `example.py` and run the following in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This command will render the animation, showing a square transforming into a circle. The `-p` flag previews the animation, and `-ql` renders it in low quality for faster processing.

### Command-Line Arguments

Manim offers various command-line arguments for customization. Key arguments include:

*   `-p`: Preview the output video automatically.
*   `-ql`: Render at a lower quality (faster rendering).
*   `-s`: Show the final frame only.
*   `-n <number>`: Skip to the nth animation of a scene.
*   `-f`: Open the output file in the file browser.

For a complete list of arguments, consult the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Resources

*   **Documentation:** [ReadTheDocs](https://docs.manim.community/)
*   **Docker:** [DockerHub](https://hub.docker.com/r/manimcommunity/manim)
*   **Jupyter Notebook Examples:** [Binder](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
*   **Gallery:** [Official gallery](https://docs.manim.community/en/stable/examples.html)

## Getting Help and Contributing

*   **Discord:** [Discord Server](https://www.manim.community/discord/)
*   **Reddit:** [Reddit Community](https://www.reddit.com/r/manim/)
*   **Contributing:**  Contributions are welcome!  See the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html).  Join the [Discord server](https://www.manim.community/discord/) to discuss potential contributions.

## How to Cite

Please cite Manim in your work by using the "cite this repository" button on the [GitHub repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Our full code of conduct and enforcement details are available on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

This software is dual-licensed under the [MIT License](http://choosealicense.com/licenses/mit/), with copyright by 3blue1brown LLC and Manim Community Developers. See [LICENSE](LICENSE) for more details.