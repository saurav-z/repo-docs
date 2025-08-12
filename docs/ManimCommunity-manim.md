<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Logo"></a>
    <br />
    <br />
</p>

# Manim: An Animation Engine for Explanatory Math Videos

**Bring your mathematical concepts to life with Manim, the powerful and versatile animation engine used to create stunning visuals, inspired by the work of 3Blue1Brown.** ([Original Repository](https://github.com/ManimCommunity/manim))

Manim, the Animation engine for explanatory math videos, is a Python library designed to help you create beautiful and engaging animations programmatically. Used extensively by creators like 3Blue1Brown, Manim empowers you to visualize complex mathematical ideas with clarity and precision.

**Key Features:**

*   **Programmatic Animation:** Create animations with Python code, giving you complete control over every detail.
*   **Precise Visualizations:** Design accurate and detailed mathematical representations, from basic shapes to complex equations.
*   **Extensive Library:** Utilize a rich set of built-in objects, transformations, and effects to bring your ideas to life.
*   **Community-Driven:** Benefit from an active and supportive community, constantly improving the software.
*   **Cross-Platform Compatibility:** Available on various operating systems.
*   **Flexible Output:** Export animations as videos or image sequences.

**Installation and Usage:**

Get started by installing ManimCE - the community-maintained version of Manim. 
*Installation instructions, and a quickstart guide are available at the official documentation site ([Documentation](https://docs.manim.community/)).*

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

Run this scene in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

**(Replace `example.py` with your filename and `SquareToCircle` with the scene name.)**

**Additional Resources:**

*   **Documentation:** Comprehensive documentation is available at [Manim Documentation](https://docs.manim.community/).
*   **Docker:** Utilize the official Manim Docker image: [Manim DockerHub](https://hub.docker.com/r/manimcommunity/manim)
*   **Jupyter Notebooks:** Experiment with Manim interactively using the Jupyter integration: [Jupyter Notebook Examples](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb)
*   **Community:** Join the [Discord Server](https://www.manim.community/discord/) or the [Reddit Community](https://www.reddit.com/r/manim/) for help and discussions.

**Contributing:**

Contributions to Manim are always welcome! Check out the [contributing guidelines](https://docs.manim.community/en/stable/contributing.html) and the [Manim dev-installation guide](https://docs.manim.community/en/latest/contributing/development.html) on how to setup your environment.

**Licensing:**

Manim is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).

**How to Cite Manim:**

Please cite Manim in your work by using the "cite this repository" button on the [GitHub repository page](https://github.com/ManimCommunity/manim). This generates a citation in your preferred format.

**Code of Conduct:**
The full code of conduct, and how it's enforced, is available on [our website](https://docs.manim.community/en/stable/conduct.html).