# GDSFactory: Design Chips with Python

**GDSFactory is a powerful Python library for designing and fabricating advanced hardware, empowering engineers and researchers to create the future of photonics, analog, quantum, MEMS, and PCBs.** ([Original Repo](https://github.com/gdsfactory/gdsfactory))

## Key Features:

*   **Intuitive Design:** Define parametric components in Python for flexible and efficient design.
*   **Seamless Simulation:** Integrate directly with popular simulation tools for accurate analysis.
*   **Comprehensive Verification:** Utilize built-in DRC (Design Rule Checking), DFM (Design for Manufacturing), and LVS (Layout Versus Schematic) for robust designs.
*   **Automated Validation:** Implement automated chip analysis and data pipelines for streamlined workflows.
*   **Versatile Output Formats:** Generate GDSII, OASIS, STL, and GERBER files for fabrication and prototyping.
*   **Extensible Architecture:** Easily add new components and functionality to tailor GDSFactory to your specific needs.

## Why Choose GDSFactory?

*   **Fast and Efficient:** Optimized for speed with the KLayout C++ library for rapid GDS operations.
*   **Open-Source and Free:** Leverage a community-driven, MIT-licensed tool without licensing fees.
*   **Thriving Community:** Join a rapidly growing ecosystem with active users, developers, and tool integrations.
*   **Open-Source Advantage:** Benefit from continuous improvements, transparency, and innovation, just like leading machine-learning libraries.

## Quick Start

Get started designing with GDSFactory in minutes.

```bash
pip install gdsfactory_install
gfi install
```

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

GDSFactory supports a wide range of open-source PDKs, enabling quick and easy design starts:

*   [Cornerstone PDK](https://github.com/gdsfactory/cspdk)
*   [SiEPIC Ebeam UBC PDK](https://github.com/gdsfactory/ubc)
*   [Quantum RF PDK](https://github.com/gdsfactory/quantum-rf-pdk)
*   [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/)
*   [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc)
*   [SkyWater 130nm CMOS PDK](https://gdsfactory.github.io/skywater130/)
*   [VTT PDK](https://github.com/gdsfactory/vtt)
*   [Luxtelligence GF PDK](https://github.com/Luxtelligence/lxt_pdk_gf)

## Foundry PDKs (NDA Required)

GDSFactory also provides access to various foundry PDKs through a **GDSFactory+** subscription, offering advanced capabilities. Visit [GDSFactory.com](https://gdsfactory.com/) to learn more.

## GDSFactory+

**GDSFactory+** offers a graphical user interface (GUI) for chip design built on top of GDSFactory and VSCode, providing access to:

*   Foundry PDKs
*   Schematic capture
*   Device and circuit simulations
*   Design verification (DRC, LVS)
*   Data analytics

## Community

Connect and collaborate with the GDSFactory community:

*   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
*   [Google Group](https://groups.google.com/g/gdsfactory)
*   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [Slack community channel](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)

## Getting Started

*   [See slides](https://docs.google.com/presentation/d/1_ZmUxbaHWo_lQP17dlT1FWX-XD8D9w7-FcuEih48d_0/edit#slide=id.g11711f50935_0_5)
*   [Read docs](https://gdsfactory.github.io/gdsfactory/)
*   [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/@gdsfactory/playlists)
*   [PIC training](https://gdsfactory.github.io/gdsfactory-photonics-training/)
*   Online course [UBCx: Silicon Photonics Design, Fabrication and Data Analysis](https://www.edx.org/learn/engineering/university-of-british-columbia-silicon-photonics-design-fabrication-and-data-ana)
*   [Visit website](https://gdsfactory.com)

## Contributors

[Insert contributor image here]

A huge thanks to all the contributors who make this project possible! We welcome all contributions – whether you're adding new features, improving documentation, or even fixing a small typo. Every contribution helps make GDSFactory better! Join us and be part of the community. 🚀

## Stargazers

[Insert stargazer chart here]