<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
    <br />
    <br />
</p>

# Manim: Create Stunning Explanatory Math Videos

**Bring your mathematical concepts to life with Manim, a powerful and versatile animation engine!**  Developed by the Manim Community and inspired by the work of 3Blue1Brown, Manim allows you to programmatically create precise and visually appealing animations for educational and explanatory videos.  ([See original repo](https://github.com/ManimCommunity/manim))

## Key Features:

*   **Programmatic Animation:** Define animations using Python code for complete control over every detail.
*   **Mathematical Visualization:** Easily create and manipulate mathematical objects, graphs, and equations.
*   **High-Quality Rendering:** Produce professional-grade videos with customizable resolutions and rendering options.
*   **Community Driven:** Actively developed and maintained by a vibrant community, ensuring continuous improvement and support.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows, with multiple installation options (pip, Docker, etc.).
*   **Integration with Jupyter:** Seamlessly integrate Manim animations into Jupyter notebooks with the `%%manim` magic command.
*   **Extensive Documentation:** Comprehensive documentation to guide you through installation, usage, and advanced features.

## Getting Started

### Installation

To get started with Manim, follow the installation instructions specific to your operating system found in the [Manim Documentation](https://docs.manim.community/en/stable/installation.html).  Alternatively, you can try Manim directly in your browser using our [online Jupyter environment](https://try.manim.community/).

### Example

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

This will render a video showing a square transforming into a circle!

### Command Line Arguments

Use command-line arguments to customize your animation:

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a faster, lower quality.
*   `-s`: Show only the final frame.
*   `-n <number>`: Skip to the nth animation in a scene.
*   `-f`: Open the output file in the file browser.

For a complete list of options, see the [Configuration Guide](https://docs.manim.community/en/stable/guides/configuration.html).

## Resources

*   **Documentation:** [https://docs.manim.community/](https://docs.manim.community/)
*   **Docker Image:** [Docker Hub](https://hub.docker.com/r/manimcommunity/manim)
*   **Jupyter Examples:** [MyBinder](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
*   **Discord:** [Discord Server](https://www.manim.community/discord/)
*   **Reddit:** [Reddit Community](https://www.reddit.com/r/manim/)

## Contributing

The Manim community welcomes contributions!  Refer to the [Contribution Guidelines](https://docs.manim.community/en/stable/contributing.html) for more information. Join the [Discord server](https://www.manim.community/discord/) to discuss contributions and stay up to date.

## How to Cite Manim

Please cite Manim in your work to acknowledge its value. Visit our [repository page](https://github.com/ManimCommunity/manim) and click the "cite this repository" button on the right sidebar to generate a citation in your preferred format.

## Code of Conduct

Our code of conduct can be found on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE) and Manim Community Developers (see LICENSE.community).