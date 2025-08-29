<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
    <br />
    <br />
</p>

# Manim: An Animation Engine for Explanatory Math Videos

**Bring your mathematical concepts to life with Manim, a powerful Python library for creating stunning animations and visualizations.** Perfect for educators, researchers, and anyone who wants to visually explain math and science, Manim allows you to create high-quality videos programmatically, like those seen in 3Blue1Brown's popular educational series.  

**[View the original repository on GitHub](https://github.com/ManimCommunity/manim)**

<br/>

## Key Features

*   **Programmatic Animation:** Create animations with precise control using Python code.
*   **Versatile Scene Creation:** Design complex scenes with various mathematical objects and transformations.
*   **High-Quality Output:** Generate videos in various formats and resolutions for professional-grade presentations.
*   **Active Community:** Benefit from a vibrant community of users and developers, with extensive documentation, examples, and support.
*   **Cross-Platform:** Run Manim on Windows, macOS, and Linux.
*   **Open Source:**  Manim is freely available under the MIT license.

## Getting Started

### Installation

To install Manim Community Edition (ManimCE), follow the instructions in the [official documentation](https://docs.manim.community/en/stable/installation.html) for your operating system.  Consider using [Docker](#docker) for an easy setup!

### Example Usage

Here's a quick example to get you started:

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

Save this code as `example.py` and run it in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will generate a video that transforms a square into a circle. Explore the [official gallery](https://docs.manim.community/en/stable/examples.html) for more advanced examples.

### Command-Line Arguments

Use command-line arguments for rendering scenes:

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality for faster results.
*   `-s`: Show the final frame only.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Open the video in the file browser.

For a complete list, consult the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Resources

*   **Documentation:** [ReadTheDocs](https://docs.manim.community/)
*   **Online Jupyter Environment:** [Try Manim Online](https://try.manim.community/)
*   **Official Gallery:** [Manim Examples](https://docs.manim.community/en/stable/examples.html)
*   **Jupyter Notebook Integration:** [JupyterLab Examples](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
*   **Discord:** [Join our Discord Server](https://www.manim.community/discord/)
*   **Reddit:** [r/manim](https://www.reddit.com/r/manim/)

## Docker

The Manim Community provides a Docker image (`manimcommunity/manim`) available on [DockerHub](https://hub.docker.com/r/manimcommunity/manim). See the [documentation](https://docs.manim.community/en/stable/installation/docker.html) for installation and usage instructions.

## Contributing

We welcome contributions to Manim! See the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html) for details.  Join our [Discord server](https://www.manim.community/discord/) for discussions and the latest updates.  To manage project dependencies, consider using `uv` as described in the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## Citation

If you use Manim in your work, please cite it by clicking the "cite this repository" button on the right sidebar on the [GitHub repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Read our full code of conduct and enforcement guidelines on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).