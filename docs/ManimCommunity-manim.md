<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
  <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"></a>
  <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
  <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
  <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit"></a>
  <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter"></a>
  <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"></a>
  <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
</p>

## Manim: Create Stunning Explanatory Math Videos with Python

**Manim** is a powerful Python library for generating mathematical animations, perfect for creating engaging educational content like the videos of [3Blue1Brown](https://www.3blue1brown.com/). ([Back to original repo](https://github.com/ManimCommunity/manim))

### Key Features:

*   **Programmatic Animation:**  Craft animations with precise control using Python code.
*   **Versatile Scene Creation:**  Build complex scenes with geometric objects, mathematical formulas, and text.
*   **Customizable Animations:**  Modify animations and customize them using a range of built-in features.
*   **Cross-Platform:**  Works seamlessly across various operating systems.
*   **Community-Driven:**  Benefit from active community support, extensive documentation, and regular updates in the Manim Community Edition (ManimCE).

### Getting Started

1.  **Installation:**
    *   Install the necessary dependencies and set up your environment by following the instructions provided in the [Documentation](https://docs.manim.community/en/stable/installation.html). You can also try out the software [in our online Jupyter environment](https://try.manim.community/).
2.  **Basic Example:**
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
    Save the code in a file called `example.py` and run the command:
    ```bash
    manim -p -ql example.py SquareToCircle
    ```

### Explore More

*   **Command Line Arguments:** Utilize a range of command line arguments to control the rendering process, e.g., `-p` for preview, `-ql` for faster, lower-quality rendering, `-s` for showing the final frame, and `-n` to skip to a specific animation. For details, visit the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).
*   **Documentation:** Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).
*   **Examples:** Find more examples at the [official gallery](https://docs.manim.community/en/stable/examples.html) and in the [GitHub repository](example_scenes).
*   **Jupyter Integration:** Use `%%manim` IPython magic to use Manim within JupyterLab and Jupyter notebooks. See [corresponding documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) and test it out [online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).
*   **Docker:**  A Docker image (`manimcommunity/manim`) is available on [DockerHub](https://hub.docker.com/r/manimcommunity/manim).  Find installation and usage instructions in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

### Get Involved

*   **Community:** Join the [Discord Server](https://www.manim.community/discord/) or the [Reddit Community](https://www.reddit.com/r/manim/) for help and discussions.
*   **Contributing:**  Contributions are welcome! See the [documentation](https://docs.manim.community/en/stable/contributing.html) for guidelines (but be aware of the ongoing refactor).
*   **Cite Manim:** When using Manim in your work, cite the repository using the "cite this repository" button on the [repository page](https://github.com/ManimCommunity/manim).

### Additional Resources

*   **Code of Conduct:** Review our code of conduct on [our website](https://docs.manim.community/en/stable/conduct.html).
*   **License:**  Manim is licensed under the MIT license, with copyright by 3blue1brown LLC and the Manim Community Developers. See [LICENSE](https://github.com/ManimCommunity/manim/blob/main/LICENSE.community) for details.