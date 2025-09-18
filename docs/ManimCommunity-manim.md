<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
    <br />
    <br />
</p>

# Manim: Create Stunning Math Animations with Python

**Transform your mathematical explanations into captivating visual stories with Manim, the open-source animation engine.** This powerful tool, inspired by the work of 3Blue1Brown, empowers you to create beautiful and precise animations for educational videos, presentations, and more. [Explore the original repository](https://github.com/ManimCommunity/manim) for more details.

<p align="center">
    <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
    <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit" ></a>
    <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter">
    <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"> </a>
    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
</p>

---

## Key Features

*   **Precise Animation Control:** Create animations frame-by-frame using Python code.
*   **Mathematical Objects:** Easily visualize mathematical concepts with built-in shapes, graphs, and equations.
*   **Customization:** Fine-tune every aspect of your animations, from colors and styles to transitions and effects.
*   **Community Driven:** Actively maintained and developed by a vibrant community.
*   **Integration:** Seamlessly integrates with Jupyter notebooks for interactive experimentation.
*   **Docker Support:** Easily set up Manim with Docker.

## Getting Started

### Installation

For detailed installation instructions and system requirements, please refer to the [official documentation](https://docs.manim.community/en/stable/installation.html).  You can also try Manim directly in your browser using [Try Manim](https://try.manim.community/).

### Example Usage

Here's a simple example of a Manim scene:

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

To render this scene, save the code as `example.py` and run the following command in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will generate a video of a square transforming into a circle. You can find more examples in the [GitHub repository](example_scenes) and the [official gallery](https://docs.manim.community/en/stable/examples.html).

## Advanced Usage

Manim offers a wide range of command-line arguments for customization.  For a complete list, consult the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Resources

*   **Documentation:** [ReadTheDocs](https://docs.manim.community/) (in progress)
*   **Docker Image:** [DockerHub](https://hub.docker.com/r/manimcommunity/manim)
*   **Jupyter Notebook Examples:** [Jupyter Examples](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)

## Getting Help

*   **Discord:** [Discord Server](https://www.manim.community/discord/)
*   **Reddit:** [Reddit Community](https://www.reddit.com/r/manim/)
*   **Issues:** Report bugs or request features by opening an issue on GitHub.

## Contributing

Contributions to Manim are welcome!  Please review the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html) for details.

Note that Manim is undergoing a refactor. New feature contributions are generally not accepted at this time. Join the [Discord server](https://www.manim.community/discord/) to stay updated on developments.

## Citing Manim

To cite Manim in your work, please use the "cite this repository" button on the [GitHub repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

See the [Code of Conduct](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is dual-licensed under the MIT license (by 3blue1brown LLC and Manim Community Developers). See [LICENSE](LICENSE) for details.