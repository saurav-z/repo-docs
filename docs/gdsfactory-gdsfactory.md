# GDSFactory: Design Chips with Python 

**GDSFactory is a powerful Python library enabling you to design chips for photonics, analog, quantum, MEMS, and PCBs.** Visit the [original repo](https://github.com/gdsfactory/gdsfactory) for the full details.

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![Downloads](https://static.pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)

GDSFactory streamlines hardware design, making it accessible and fun by converting your Python code into industry-standard CAD files (GDS, OASIS, STL, GERBER).

![cad](https://i.imgur.com/3cUa2GV.png)

## Key Features

*   **Intuitive Design:** Define components using Python, making hardware design programmable and flexible.
*   **Versatile Output:** Generate industry-standard CAD files (GDSII, OASIS, STL, GERBER) for fabrication and 3D printing.
*   **Integrated Simulation:** Seamlessly integrates with major simulation tools for accurate design verification.
*   **Comprehensive Verification:** Built-in DRC (Design Rule Checking), DFM (Design for Manufacturing), and LVS (Layout Versus Schematic) capabilities ensure design integrity.
*   **Automated Validation:** Create automated workflows for chip analysis and data pipelines, streamlining post-fabrication testing.
*   **Extensible and Customizable:** Easily add new components and functionalities to tailor GDSFactory to your specific needs.

## Quick Start

Get started with GDSFactory in just a few steps:

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

## Why Choose GDSFactory?

*   **Speed and Efficiency:** Designed for speed, GDSFactory leverages the KLayout C++ library for fast GDS operations.
*   **Open Source Advantage:** Benefit from a growing community, continuous contributions, and transparent development, just like leading machine-learning libraries.
*   **Thriving Ecosystem:** Join a community of users and developers with extensive integrations and resources.
*   **Flexibility & Extensibility:** Easily adapt GDSFactory to meet your unique project needs with its flexible Python interface.

## PDKs

GDSFactory offers both open-source and foundry-specific Process Design Kits (PDKs).

### Open-Source PDKs (No NDA Required)

*   [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/)
*   [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc)
*   [SkyWater 130nm CMOS PDK](https://gdsfactory.github.io/skywater130/)
*   [VTT PDK](https://github.com/gdsfactory/vtt)
*   [Cornerstone PDK](https://github.com/gdsfactory/cspdk)
*   [Luxtelligence GF PDK](https://github.com/Luxtelligence/lxt_pdk_gf)

### Foundry PDKs (NDA Required)

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

**GDSFactory+** provides a Graphical User Interface (GUI) for chip design, built on top of GDSFactory and VSCode. Benefits of GDSFactory+ include:

*   Foundry PDK access
*   Schematic capture
*   Device and circuit Simulations
*   Design verification (DRC, LVS)
*   Data analytics

## Community and Resources

*   **Documentation:** [GDSFactory Documentation](https://gdsfactory.github.io/gdsfactory/)
*   **Video Tutorials:** [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/@gdsfactory/playlists)
*   **Discussions:** [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
*   **Google Group:** [Google Group](https://groups.google.com/g/gdsfactory)
*   **LinkedIn:** [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   **Slack:** [Slack community channel](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)

## Who Uses GDSFactory?

GDSFactory is used by many organizations around the world, including:

![logos](https://i.imgur.com/VzLNMH1.png)

> *"I've used **GDSFactory** since 2017 for all my chip tapeouts. I love that it is fast, easy to use, and easy to extend. It's the only tool that allows us to have an end-to-end chip design flow (design, verification and validation)."*
>
> -- Joaquin Matres - **Google**

## Contributors

A huge thanks to all the contributors who make this project possible!
![contributors](https://i.imgur.com/0AuMHZE.png)