<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
    <br />
    <br />
</p>

<!-- Badges Section -->
<p align="center">
  <a href="https://pypi.org/project/manim/">
    <img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release">
  </a>
  <a href="https://hub.docker.com/r/manimcommunity/manim">
    <img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image">
  </a>
  <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb">
    <img src="https://mybinder.org/badge_logo.svg" alt="Binder">
  </a>
  <a href="http://choosealicense.com/licenses/mit/">
    <img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License">
  </a>
  <a href="https://www.reddit.com/r/manim/">
    <img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit">
  </a>
  <a href="https://twitter.com/manim_community/">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter">
  </a>
  <a href="https://www.manim.community/discord/">
    <img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
  </a>
  <a href="https://docs.manim.community/">
    <img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status">
  </a>
  <a href="https://pepy.tech/project/manim">
    <img src="https://pepy.tech/badge/manim/month?" alt="Downloads">
  </a>
  <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
</p>

## Manim: Create Stunning Explanatory Math Videos with Python

Manim is a powerful Python library for creating dynamic and visually engaging mathematical animations, bringing your ideas to life with code, similar to those seen in 3Blue1Brown's educational videos. Explore the [Manim Community Repository](https://github.com/ManimCommunity/manim) for the source code.

**Key Features:**

*   **Code-Based Animation:** Animate mathematical concepts and visuals using Python code for precise control.
*   **Versatile Scene Creation:** Design complex animations with various shapes, transformations, and effects.
*   **Community-Driven Development:** Benefit from an active community, regular updates, and extensive documentation.
*   **Interactive Exploration:** Experiment with Manim in your browser using the [online Jupyter environment](https://try.manim.community/).
*   **Cross-Platform Support:** Available on Linux, Windows, and macOS.
*   **Docker Image:** Easily set up a Manim environment using the official Docker image.
*   **Jupyter Integration:** Utilize the `%%manim` IPython magic for seamless integration within JupyterLab notebooks.

**Installation**

For detailed installation instructions, refer to the [official documentation](https://docs.manim.community/en/stable/installation.html).

**Usage**

Create engaging scenes by defining animations programmatically.

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

Save this example as `example.py` and run:

```bash
manim -p -ql example.py SquareToCircle
```

**Command-Line Arguments**

Manim offers a variety of command-line arguments to control rendering and output:

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality for faster rendering.
*   `-s`: Skip to the final frame.
*   `-n <number>`: Skip to the nth animation in a scene.
*   `-f`: Show the output file in the file browser.

For a complete list, see the [configuration guide](https://docs.manim.community/en/stable/guides/configuration.html).

**Resources:**

*   **Documentation:** Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).
*   **Docker:** Find the Manim Docker image on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).
*   **Community Support:** Get help and connect with other users on [Discord](https://www.manim.community/discord/) or [Reddit](https://www.reddit.com/r/manim/).

**Contributing**

Contributions are welcome! Review the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html), but be aware that a major refactor is underway. Join the [Discord server](https://www.manim.community/discord/) for the latest updates.

**How to Cite Manim**

If you use Manim in your work, please cite the repository using the citation information provided on the [GitHub repository page](https://github.com/ManimCommunity/manim).

**Code of Conduct**

Read our full code of conduct on [our website](https://docs.manim.community/en/stable/conduct.html).

**License**

Manim is licensed under the MIT license, with copyright by 3blue1brown LLC and the Manim Community Developers (see LICENSE).