<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png"></a>
    <br />
    <br />
</p>

# Manim: Create Stunning Math Animations with Python

**Bring your mathematical concepts to life with Manim, the powerful Python animation engine used by 3Blue1Brown and the ManimCommunity!**

**[Explore the Original Repository on GitHub](https://github.com/ManimCommunity/manim)**

## Key Features:

*   **Programmatic Animation:** Create precise and customizable animations using Python code.
*   **Versatile Scene Building:** Construct scenes with geometric objects, mathematical formulas, and text.
*   **High-Quality Output:** Render videos in various resolutions and formats for professional-looking results.
*   **Active Community:** Benefit from a vibrant and supportive community, resources, and examples.
*   **Cross-Platform Compatibility:** Works on Linux, macOS, and Windows.
*   **Interactive Workflow:** Utilize the `%%manim` IPython magic for seamless integration with Jupyter notebooks.
*   **Docker Support:** Easily set up and run Manim with Docker.

## What is Manim?

Manim is an animation engine specifically designed for generating explanatory math videos. It allows you to programmatically create animations of mathematical concepts, geometric shapes, and other visual elements.  This community edition (ManimCE) builds upon the original work of Grant Sanderson, the creator of the popular 3Blue1Brown YouTube channel, with added features, improved documentation, and ongoing community support.

## Getting Started

### Installation

Detailed installation instructions are available in the [Manim Documentation](https://docs.manim.community/en/stable/installation.html). This includes information for various operating systems and installation methods.  A [Jupyter environment](https://try.manim.community/) is available to test Manim without installing locally.

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

Save the code as `example.py` and run the following command in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will generate a video showing a square transforming into a circle.  Explore the [examples](https://docs.manim.community/en/stable/examples.html) in the official documentation and the [GitHub repository](https://github.com/ManimCommunity/manim) to find more examples.

### Command-Line Arguments

Customize your rendering process using command-line arguments. Some useful options include:

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality for faster processing.
*   `-s`: Skip to the final frame.
*   `-n <number>`: Skip to the nth animation in a scene.
*   `-f`: Show the file in the file browser.

Find a complete list of available command-line arguments in the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Resources and Support

*   **Documentation:**  Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).
*   **Docker:**  Use the community-maintained Docker image: [`manimcommunity/manim`](https://hub.docker.com/r/manimcommunity/manim).
*   **Community:** Get help and connect with other users via the [Discord Server](https://www.manim.community/discord/) and the [Reddit Community](https://www.reddit.com/r/manim/).

## Contributing

Contributions to Manim are welcome! Please review the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html) for more information.

**Important Note:** Manim is undergoing a significant refactor, so new feature implementations are currently discouraged.  Join the Discord server to stay updated and discuss potential contributions.  Project developers recommend using `uv` for package management.

## Citing Manim

To properly credit Manim in your work, please use the "cite this repository" button on the [GitHub repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Please review the [Code of Conduct](https://docs.manim.community/en/stable/conduct.html) to ensure a positive and respectful community environment.

## License

Manim is dual-licensed under the MIT license, with copyright by 3blue1brown LLC and Manim Community Developers.  See the `LICENSE` and `LICENSE.community` files for details.
```
Key improvements and SEO considerations:

*   **Clear Hook:**  A strong, concise opening sentence that grabs attention and states the primary benefit.
*   **Keyword Optimization:** Incorporated relevant keywords throughout the text (e.g., "math animations," "Python animation engine," "3Blue1Brown").
*   **Headings and Structure:** Used clear, descriptive headings and subheadings for readability and SEO.
*   **Bulleted Key Features:**  Highlights the main advantages of using Manim in an easy-to-scan format.
*   **Internal Linking:**  Links to other relevant parts of the README, documentation, and examples.
*   **Call to Action:** Encourages users to explore the repository, get started, and contribute.
*   **Concise and Informative:** Replaced some longer paragraphs with more concise statements.
*   **SEO-Friendly Formatting:**  Used Markdown headings, bold text, and lists for readability and SEO.
*   **Community Focus:** Emphasizes the community aspect and support.
*   **Docker and Jupyter Integration:**  Highlights important features.
*   **Consistent Tone:**  Maintained a professional and enthusiastic tone.
*   **Concise Summarization:**  Condensed the information without losing key details.