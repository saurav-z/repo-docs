# GDSFactory: Design Chips with Python and Open-Source Power

**GDSFactory is a powerful Python library revolutionizing chip design for photonics, analog, quantum, MEMS, PCBs, and 3D-printable objects, enabling you to design, simulate, and fabricate hardware with ease.** [Check out the original repo](https://github.com/gdsfactory/gdsfactory)

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![Downloads](https://static.pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)

<br>

## Key Features

*   **Intuitive Python-Based Design:** Define components using Python code for flexible and parametric design.
*   **Comprehensive Design Flow:** Streamline your workflow with design, verification, and validation tools.
*   **Simulation Integration:** Direct interfaces with leading simulation tools for accurate analysis.
*   **Verification Capabilities:** Includes DRC (Design Rule Checking), DFM (Design for Manufacturing), and LVS (Layout Versus Schematic) for robust designs.
*   **Automated Validation:** Simplify chip analysis with automated test protocols and data pipelines.
*   **Multiple Output Formats:** Generate industry-standard files like GDSII, OASIS, STL, and GERBER for fabrication.
*   **Open-Source PDKs:** Leverage publicly available PDKs for various foundries.

**Benefits of GDSFactory:**

*   **Efficiency:** Achieve faster design cycles with Python's power and KLayout's performance.
*   **Extensibility:** Easily add new components and customize functionality.
*   **Community-Driven:** Benefit from a thriving open-source ecosystem.
*   **Cost-Effective:** Reduce design costs with a free and open-source solution.
<br>

## Quick Start

Get started by installing GDSFactory:

```bash
pip install gdsfactory_install
gfi install
```

Here's a basic example:

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

## Open-Source PDKs (No NDA Required)

*   [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/)
*   [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc)
*   [SkyWater 130nm CMOS PDK](https://gdsfactory.github.io/skywater130/)
*   [VTT PDK](https://github.com/gdsfactory/vtt)
*   [Cornerstone PDK](https://gdsfactory.github.io/cspdk)
*   [Luxtelligence GF PDK](https://github.com/Luxtelligence/lxt_pdk_gf)

## Foundry PDKs (NDA Required)

Access to the following PDKs requires a **GDSFactory+** subscription.
To sign up, visit [GDSFactory.com](https://gdsfactory.com/).

Available PDKs under NDA:

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

## GDSFactory+

**GDSFactory+** offers a Graphical User Interface for chip design, built on top of GDSFactory and VSCode. It provides you:

*   Foundry PDK access
*   Schematic capture
*   Device and circuit Simulations
*   Design verification (DRC, LVS)
*   Data analytics

## Getting Started

*   [See slides](https://docs.google.com/presentation/d/1_ZmUxbaHWo_lQP17dlT1FWX-XD8D9w7-FcuEih48d_0/edit#slide=id.g11711f50935_0_5)
*   [Read docs](https://gdsfactory.github.io/gdsfactory/)
*   [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/@gdsfactory/playlists)
*   See announcements on [GitHub](https://github.com/gdsfactory/gdsfactory/discussions/547), [google-groups](https://groups.google.com/g/gdsfactory) or [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=250169028)
*   [PIC training](https://gdsfactory.github.io/gdsfactory-photonics-training/)
*   Online course [UBCx: Silicon Photonics Design, Fabrication and Data Analysis](https://www.edx.org/learn/engineering/university-of-british-columbia-silicon-photonics-design-fabrication-and-data-ana), where students can use GDSFactory to create a design, have it fabricated, and tested.
*   [Visit website](https://gdsfactory.com)

## Who is Using GDSFactory?

[<img src="https://i.imgur.com/VzLNMH1.png" alt="Logos of companies using GDSFactory" width="600"/>](https://gdsfactory.com/)

> "I've used **GDSFactory** since 2017 for all my chip tapeouts. I love that it is fast, easy to use, and easy to extend. It's the only tool that allows us to have an end-to-end chip design flow (design, verification and validation)."
>
> <div style="text-align: right; margin-right: 10%;">Joaquin Matres - <strong>Google</strong></div>

---

> "I've relied on **GDSFactory** for several tapeouts over the years. It's the only tool I've found that gives me the flexibility and scalability I need for a variety of projects."
>
> <div style="text-align: right; margin-right: 10%;">Alec Hammond - <strong>Meta Reality Labs Research</strong></div>

---

> "The best photonics layout tool I've used so far and it is leaps and bounds ahead of any commercial alternatives out there. Feels like GDSFactory is freeing photonics."
>
> <div style="text-align: right; margin-right: 10%;">Hasitha Jayatilleka - <strong>LightIC Technologies</strong></div>

---

> "As an academic working on large scale silicon photonics at CMOS foundries I've used GDSFactory to go from nothing to full-reticle layouts rapidly (in a few days). I particularly appreciate the full-system approach to photonics, with my layout being connected to circuit simulators which are then connected to device simulators. Moving from legacy tools such as gdspy and phidl to GDSFactory has sped up my workflow at least an order of magnitude."
>
> <div style="text-align: right; margin-right: 10%;">Alex Sludds - <strong>MIT</strong></div>

---

> "I use GDSFactory for all of my photonic tape-outs. The Python interface makes it easy to version control individual photonic components as well as entire layouts, while integrating seamlessly with KLayout and most standard photonic simulation tools, both open-source and commercial.
>
> <div style="text-align: right; margin-right: 10%;">Thomas Dorch - <strong>Freedom Photonics</strong></div>

## Why Use GDSFactory?

*   **Fast, Extensible, and Easy to Use**: Designed for efficient and flexible chip design.
*   **Open-Source & Free:** No licensing fees; modify and extend it freely.
*   **Growing Ecosystem:** The most popular EDA tool with a vibrant community.
*   **Performance:** Powered by the KLayout C++ library, ensuring speed in GDS handling.

**GDSFactory vs. Other Tools - Performance Benchmarks**

| Benchmark      |  gdspy  | GDSFactory | Gain |
| :------------- | :-----: | :--------: | :--: |
| 10k_rectangles | 80.2 ms |  4.87 ms   | 16.5 |
| boolean-offset | 187 μs  |  44.7 μs   | 4.19 |
| bounding_box   | 36.7 ms |   170 μs   | 216  |
| flatten        | 465 μs  |  8.17 μs   | 56.9 |
| read_gds       | 2.68 ms |   94 μs    | 28.5 |

## Contributors

A huge thanks to all the contributors who make this project possible!

We welcome all contributions—whether you're adding new features, improving documentation, or even fixing a small typo. Every contribution helps make GDSFactory better!

Join us and be part of the community. 🚀

![contributors](https://i.imgur.com/0AuMHZE.png)

## Stargazers

[![Stargazers over time](https://starchart.cc/gdsfactory/gdsfactory.svg)](https://starchart.cc/gdsfactory/gdsfactory)

## Community

Connect with the GDSFactory community:
*   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
*   [Google Group](https://groups.google.com/g/gdsfactory)
*   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [Slack community channel](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)