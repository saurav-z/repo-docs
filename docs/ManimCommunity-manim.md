<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png"></a>
    <br />
    <br />
</p>

# Manim: Create Stunning Math Animations with Code

**Bring your mathematical ideas to life with Manim, the powerful and versatile animation engine for creating explanatory math videos.  Learn more and contribute at the [official Manim Community repository](https://github.com/ManimCommunity/manim).**

Manim empowers you to programmatically generate precise and visually appealing animations, making complex concepts easy to understand.  This is the community edition, a fork of the original created by 3Blue1Brown.

**Key Features:**

*   **Code-Driven Animation:**  Define animations using Python code for complete control.
*   **Mathematical Objects:** Easily represent and manipulate mathematical objects like graphs, equations, and geometric shapes.
*   **Versatile Output:** Create videos in various formats and resolutions.
*   **Active Community:** Benefit from a vibrant community, extensive documentation, and examples.
*   **Jupyter Notebook Integration:** Utilize the `%%manim` magic command to use the tool seamlessly in Jupyter notebooks.

**Key Links:**
* [Documentation](https://docs.manim.community/)
* [Example Gallery](https://docs.manim.community/en/stable/examples.html)
* [Discord Server](https://www.manim.community/discord/)
* [Reddit Community](https://www.reddit.com/r/manim/)
* [DockerHub](https://hub.docker.com/r/manimcommunity/manim)

## Installation

For detailed installation instructions, tailored to your operating system, please refer to the [official documentation](https://docs.manim.community/en/stable/installation.html). Consider using the online Jupyter environment at [try.manim.community](https://try.manim.community/) for initial exploration.

## Usage

Here's a basic example of a Manim scene:

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

Save this code in a file (e.g., `example.py`) and run the following command in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

This will generate a video showing a square transforming into a circle.

## Command-Line Arguments

Manim offers several command-line arguments to customize your animations:

*   `-p`: Preview the video after rendering.
*   `-ql`: Render at a lower quality for faster processing.
*   `-s`: Show only the final frame.
*   `-n <number>`: Skip to the nth animation.
*   `-f`: Show the file in the file browser.

For a comprehensive list, consult the [configuration documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Contributing

We welcome contributions to Manim!  Refer to the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html) and join our [Discord server](https://www.manim.community/discord/) to discuss any potential contributions.

## How to Cite Manim

To properly cite Manim, use the citation information available on the [GitHub repository page](https://github.com/ManimCommunity/manim) under the "Cite this repository" button.

## License

Manim is double-licensed under the [MIT license](https://github.com/ManimCommunity/manim/blob/main/LICENSE) with copyright by 3blue1brown LLC and Manim Community Developers.