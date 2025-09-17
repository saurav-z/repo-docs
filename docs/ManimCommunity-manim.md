<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community"></a>
</p>

<p align="center">
    <i>Craft Stunning Explanatory Math Videos with Ease</i>
</p>

<p align="center">
    <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
    <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit"></a>
    <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter"></a>
    <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
    <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"> </a>
    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
</p>

---

## Manim: The Animation Engine for Engaging Math Videos

Manim is a powerful Python library designed for creating mathematical animations, perfect for educational content and visual explanations.  It's the tool behind the captivating videos of [3Blue1Brown](https://www.3blue1brown.com/).  This community edition, ManimCE, is actively maintained, providing improved features and a vibrant community.  Explore its capabilities and create your own stunning visuals!

**Key Features:**

*   **Programmatic Animation:**  Create precise animations using Python code for ultimate control.
*   **Versatile Scene Construction:** Build complex scenes with geometric objects, mathematical expressions, and dynamic transformations.
*   **Command-Line Tools:**  Utilize a robust command-line interface for rendering and customizing animations.
*   **Jupyter Integration:** Seamlessly integrate Manim into Jupyter notebooks for interactive exploration and experimentation.
*   **Extensive Documentation:**  Benefit from comprehensive documentation to guide you through the library's features.
*   **Active Community:**  Join a thriving community for support, collaboration, and inspiration.

## Getting Started

### Installation

To get started with Manim, follow the installation instructions in the [Documentation](https://docs.manim.community/en/stable/installation.html) for your operating system. If you'd like to try it without installing, check out our [online Jupyter environment](https://try.manim.community/).

### Example

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

Save this code as `example.py` and run the following in your terminal:

```bash
manim -p -ql example.py SquareToCircle
```

### Command-Line Arguments

Manim uses command-line arguments for various tasks.  The `-p` flag previews your video, and `-ql` renders at a lower quality. For a full list, see the [documentation](https://docs.manim.community/en/stable/guides/configuration.html).

## Resources

*   **[Documentation](https://docs.manim.community/):** Comprehensive guides and API reference.
*   **[Docker](https://hub.docker.com/r/manimcommunity/manim):**  Utilize a pre-built Docker image.
*   **[Discord](https://www.manim.community/discord/):**  Get help and connect with the community.
*   **[Reddit](https://www.reddit.com/r/manim/):**  Share your creations and ask questions.
*   **[GitHub Repository](https://github.com/ManimCommunity/manim):** The official repository.

## Contributing

Contributions are welcome!  Check the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html).  Please note that Manim is currently undergoing a major refactor. Join the [Discord server](https://www.manim.community/discord/) for the latest updates.

## How to Cite Manim

Cite Manim in your work using the "cite this repository" button on the [GitHub repository page](https://github.com/ManimCommunity/manim).

## Code of Conduct

Read our [Code of Conduct](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is double-licensed under the MIT license (3blue1brown LLC & Manim Community Developers). See the [LICENSE](https://github.com/ManimCommunity/manim/blob/main/LICENSE) files.