<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

# Manim: Create Stunning Math Animations with Code

**Bring your math concepts to life with Manim, a powerful animation engine that lets you create explanatory math videos using Python!**  This is the community edition of Manim, a fork of the original project by 3Blue1Brown, offering continuous development, improved features, and a supportive community.  [Explore the original project](https://github.com/ManimCommunity/manim) for inspiration.

## Key Features:

*   **Programmatic Animation:**  Create precise and customizable animations using Python code.
*   **Versatile Scene Creation:**  Build complex scenes with shapes, text, equations, and more.
*   **Mathematical Visualization:** Visualize mathematical concepts in a clear and engaging way.
*   **Community Driven:** Benefit from active community support, ongoing development, and extensive documentation.
*   **Cross-Platform:** Runs on various operating systems, including Windows, macOS, and Linux.
*   **Jupyter Integration:** Use Manim directly within Jupyter notebooks for interactive experimentation.

## Getting Started

### Installation

ManimCE, the community edition, has dependencies that must be installed. Check out the [documentation](https://docs.manim.community/en/stable/installation.html) for detailed installation instructions.  You can also try out Manim directly in a web browser using our [online Jupyter environment](https://try.manim.community/).

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

Save this code as `example.py` and run the following command in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will render and preview a scene transforming a square into a circle.

### Command Line Arguments

Manim offers a variety of command-line arguments for customization. Explore these:

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality for faster processing.
*   `-s`: Show only the final frame.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Open the file in the file browser.

For a comprehensive list, refer to the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Resources

*   **Documentation:** [ReadTheDocs](https://docs.manim.community/)
*   **Docker:** [DockerHub](https://hub.docker.com/r/manimcommunity/manim) - Docker image for easy setup.
*   **Examples:** [Official Gallery](https://docs.manim.community/en/stable/examples.html) - Browse and learn from various animation examples.
*   **Jupyter Notebook Example:** [Binder](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
*   **Community:**
    *   [Discord Server](https://www.manim.community/discord/) - Get help and interact with the community.
    *   [Reddit Community](https://www.reddit.com/r/manim/) - Discuss and share your creations.

## Contributing

Contributions are welcome!  See the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html) and join our [Discord server](https://www.manim.community/discord/) to discuss potential contributions.  Note that new features will be limited during the major refactor period.

## Citing Manim

To acknowledge Manim in your work, please cite this repository using the "cite this repository" button on the right sidebar of the [GitHub page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Read the code of conduct and enforcement on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).