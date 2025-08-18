<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
</p>

# Manim: Create Stunning Math Animations

**Bring your mathematical concepts to life with Manim, the powerful animation engine used by 3Blue1Brown and the Manim Community!**  This is the community-maintained version, offering enhanced features, documentation, and a vibrant community.  [Explore the original repository](https://github.com/ManimCommunity/manim).

## Key Features:

*   **Programmatic Animation:** Create precise animations using Python code.
*   **Versatile Scene Creation:** Design complex scenes with shapes, text, and mathematical formulas.
*   **High-Quality Output:** Render videos in various resolutions for professional-looking results.
*   **Community-Driven:** Benefit from active development, regular updates, and a supportive community.
*   **Jupyter Notebook Integration:** Use the `%%manim` magic command for seamless animation creation within Jupyter environments.
*   **Docker Support:** Easily set up and run Manim using pre-built Docker images.

## Core Features:

*   **Precise Control:** Fine-tune every aspect of your animations with code.
*   **Mathematical Symbols:** Easily integrate and animate mathematical notation.
*   **Scene Organization:** Structure your animations logically with scenes and animations.
*   **Extensible Library:** Expand Manim's functionality with custom objects and animations.

## Get Started

### Installation

For detailed installation instructions, including system requirements and platform-specific guides, please refer to the [official documentation](https://docs.manim.community/en/stable/installation.html). You can also try Manim directly in a browser using our [online Jupyter environment](https://try.manim.community/).

### Example Usage

Here's a basic example to get you started:

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

Save the code in a file (e.g., `example.py`) and render it using the command line:

```bash
manim -p -ql example.py SquareToCircle
```

(This will preview your scene in a window and render at a quick, low quality)

### Command Line Arguments

Manim offers several command-line arguments for controlling rendering behavior. Key flags include:

*   `-p`: Preview the output video automatically.
*   `-ql`: Render at a lower quality for faster processing.
*   `-s`: Show the final frame only (skip the animation).
*   `-n <number>`: Skip ahead to the nth animation in a scene.
*   `-f`: Open the file in the file browser after rendering.

For a comprehensive list, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Resources

*   **Documentation:** Comprehensive guides and tutorials are available at [ReadTheDocs](https://docs.manim.community/).
*   **Docker:** Use the `manimcommunity/manim` Docker image.  [Learn more](https://docs.manim.community/en/stable/installation/docker.html).
*   **Examples:** Explore a wide range of examples in the [official gallery](https://docs.manim.community/en/stable/examples.html) and in the [GitHub repository](example_scenes).
*   **Jupyter Notebooks:** Utilize Manim's `%%manim` IPython magic to use Manim in Jupyter environments, as demonstrated [here](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).
*   **Community:** Get help and connect with other users via our [Discord Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim/).

## Contributing

We encourage contributions to Manim! Please consult the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html) for details.  Join our [Discord server](https://www.manim.community/discord/) to discuss potential contributions and stay up-to-date on the project.  Note: During the current refactor, new features will generally not be accepted.

Most developers use `uv` for management. You'll want to have uv installed and available in your environment.
Learn more about `uv` at its [documentation](https://docs.astral.sh/uv/) and find out how to install manim with uv at the [manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) in the manim documentation.

## How to Cite Manim

To properly acknowledge Manim in your work, please cite the repository using the "cite this repository" button on the right sidebar of the [GitHub repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Our Code of Conduct is available on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is dual-licensed under the [MIT license](https://github.com/ManimCommunity/manim/blob/main/LICENSE) with copyright by 3blue1brown LLC, and the [MIT license](https://github.com/ManimCommunity/manim/blob/main/LICENSE.community) with copyright by Manim Community Developers.