<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/main/logo/cropped.png" alt="Manim Community Logo"></a>
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

# Manim: Create Stunning Explanatory Math Videos

**Manim is a powerful and versatile animation engine that allows you to programmatically generate high-quality videos perfect for explaining mathematical concepts.** ([See the original repo](https://github.com/ManimCommunity/manim)).

## Key Features

*   **Programmatic Animation:** Create animations with code, providing precision and control.
*   **Mathematical Visualization:** Easily visualize mathematical objects and concepts.
*   **Customizable Scenes:** Design and build complex scenes to explain your ideas.
*   **Community Driven:** Benefit from an active community, detailed documentation, and ongoing development.
*   **Integration:** Seamlessly integrates with Jupyter notebooks for interactive use.
*   **Cross-Platform:** Works on various operating systems through local installation and Docker.

## Getting Started

### Installation

To install the Community Edition of Manim (ManimCE), follow the instructions in the [documentation](https://docs.manim.community/en/stable/installation.html). This version offers continued development, improved features, and an active community, distinct from the original 3Blue1Brown version. You can also try Manim online using [our online Jupyter environment](https://try.manim.community/).

### Basic Usage Example

Here's a quick example of how to create a simple animation:

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

This command renders and previews a scene where a square transforms into a circle.

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://docs.manim.community/).

## Docker

Use Docker to easily run Manim. The Docker image is available at [DockerHub](https://hub.docker.com/r/manimcommunity/manim). Instructions on how to install and use the Docker image can be found in the [documentation](https://docs.manim.community/en/stable/installation/docker.html).

## Resources and Community Support

*   **Discord Server:** Get help and engage with the community on our [Discord Server](https://www.manim.community/discord/).
*   **Reddit Community:** Join the discussion on the [Reddit Community](https://www.reddit.com/r/manim/).
*   **Submit Issues:** Report bugs or suggest features by opening an issue on [GitHub](https://github.com/ManimCommunity/manim).

## Contributing

We welcome contributions! Please review the [contribution guidelines](https://docs.manim.community/en/stable/contributing.html) before submitting.  We recommend joining our [Discord server](https://www.manim.community/discord/) to discuss any potential contributions and stay updated on the latest developments, especially during our refactor phase.

## How to Cite Manim

To cite Manim, please go to our [repository page](https://github.com/ManimCommunity/manim) and use the "cite this repository" button to generate a citation in your preferred format.

## Code of Conduct

Our code of conduct is detailed on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

Manim is licensed under the MIT license (copyright by 3blue1brown LLC and Manim Community Developers).  See the [LICENSE](https://github.com/ManimCommunity/manim/blob/main/LICENSE) file for more details.