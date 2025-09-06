# GDSFactory: Design Chips with Python 

**GDSFactory is a powerful Python library enabling you to design cutting-edge chips for photonics, analog, quantum, MEMS, and more.** ([See Original Repo](https://github.com/gdsfactory/gdsfactory))

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![Downloads](https://static.pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)

GDSFactory empowers engineers and designers to create complex hardware designs intuitively and efficiently. It transforms Python code into industry-standard CAD files like GDSII, OASIS, STL, and GERBER.

![cad](https://i.imgur.com/3cUa2GV.png)

## Key Features

*   **Design Automation**: Define parametric components in Python, making design changes and iterations easy.
*   **Multi-Discipline Support**: Perfect for photonics, analog, quantum, MEMS, and PCB design.
*   **Integrated Simulation**:  Seamlessly integrates with popular simulation tools, streamlining your workflow.
*   **Verification & Validation**: Includes DRC, DFM, LVS, and automated chip analysis for robust design.
*   **Flexible Output**: Generates industry-standard GDSII, OASIS, STL, and GERBER files.
*   **Open-Source & Extensible**:  Built on open-source principles, allowing for customization and community contributions.

## Quick Start

Get started by installing the library:

```bash
pip install gdsfactory_install
gfi install
```

Here's a simple example:

```python
import gdsfactory as gf

# Create a new component
c = gf.Component()

# Add a rectangle
r = gf.components.rectangle(size=(10, 10), layer=(1, 0))
rect = c.add_ref(r)

# Add text elements
t1 = gf.components.text("Hello", size=10, layer=(2, 0))
t2 = gf.components.text("world", size=10, layer=(2, 0))

text1 = c.add_ref(t1)
text2 = c.add_ref(t2)

# Position elements
text1.xmin = rect.xmax + 5
text2.xmin = text1.xmax + 2
text2.rotate(30)

# Show the result
c.show()
```

## Why Use GDSFactory?

*   **Speed & Efficiency:**  GDSFactory leverages the KLayout C++ library for exceptionally fast performance.
*   **Open Source Advantage:** Benefit from community contributions, transparency, and continuous innovation.
*   **Growing Ecosystem:** Join a thriving community with extensive tool integrations.

## Open-Source PDKs (No NDA Required)

Access a wide range of open-source PDKs:

*   [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/)
*   [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc)
*   [SkyWater 130nm CMOS PDK](https://gdsfactory.github.io/skywater130/)
*   [VTT PDK](https://github.com/gdsfactory/vtt)
*   [Cornerstone PDK](https://gdsfactory.github.io/cspdk)
*   [Luxtelligence GF PDK](https://github.com/Luxtelligence/lxt_pdk_gf)

## Foundry PDKs (NDA Required)

Explore advanced PDKs by subscribing to **GDSFactory+**:

*   AIM Photonics
*   AMF Photonics
*   CompoundTek Photonics
*   Fraunhofer HHI Photonics
*   Smart Photonics
*   Tower Semiconductor PH18
*   Tower PH18DA by OpenLight
*   III-V Labs
*   LioniX
*   Ligentec
*   Lightium
*   Quantum Computing Inc. (QCI)

Learn more about GDSFactory+ at [GDSFactory.com](https://gdsfactory.com/).

## Performance Benchmarks

GDSFactory provides significant performance gains compared to other tools:

| Benchmark      |  gdspy  | GDSFactory | Gain |
| :------------- | :-----: | :--------: | :--: |
| 10k_rectangles | 80.2 ms |  4.87 ms   | 16.5 |
| boolean-offset | 187 μs  |  44.7 μs   | 4.19 |
| bounding_box   | 36.7 ms |   170 μs   | 216  |
| flatten        | 465 μs  |  8.17 μs   | 56.9 |
| read_gds       | 2.68 ms |   94 μs    | 28.5 |

## Who is using GDSFactory?

Join the growing number of organizations benefiting from GDSFactory:

![logos](https://i.imgur.com/VzLNMH1.png)

> "I've used **GDSFactory** since 2017 for all my chip tapeouts. I love that it is fast, easy to use, and easy to extend. It's the only tool that allows us to have an end-to-end chip design flow (design, verification and validation)."
>
> Joaquin Matres - **Google**

...and many more.

## Community

Connect and contribute to the GDSFactory community:

*   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
*   [Google Group](https://groups.google.com/g/gdsfactory)
*   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [Slack community channel](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)

## Getting Started

*   [See slides](https://docs.google.com/presentation/d/1_ZmUxbaHWo_lQP17dlT1FWX-XD8D9w7-FcuEih48d_0/edit#slide=id.g11711f50935_0_5)
*   [Read docs](https://gdsfactory.github.io/gdsfactory/)
*   [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/@gdsfactory/playlists)
*   See announcements on [GitHub](https://github.com/gdsfactory/gdsfactory/discussions/547), [google-groups](https://groups.google.com/g/gdsfactory) or [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=250169028)
*   [PIC training](https://gdsfactory.github.io/gdsfactory-photonics-training/)
*   Online course [UBCx: Silicon Photonics Design, Fabrication and Data Analysis](https://www.edx.org/learn/engineering/university-of-british-columbia-silicon-photonics-design-fabrication-and-data-ana), where students can use GDSFactory to create a design, have it fabricated, and tested.
*   [Visit website](https://gdsfactory.com)

## Contributors

A huge thanks to all the contributors who make this project possible!

Join us and be part of the community. 🚀

![contributors](https://i.imgur.com/0AuMHZE.png)

## Stargazers

[![Stargazers over time](https://starchart.cc/gdsfactory/gdsfactory.svg)](https://starchart.cc/gdsfactory/gdsfactory)