# GDSFactory: Design Chips with Python for Photonics, Analog, Quantum, and Beyond

**GDSFactory is a powerful Python library that empowers you to design and fabricate chips (Photonics, Analog, Quantum, MEMS) with code, making hardware design accessible and fun.**  For the original repository, visit: [https://github.com/gdsfactory/gdsfactory](https://github.com/gdsfactory/gdsfactory).

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![Downloads](https://static.pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)

## Key Features

*   **Python-Based Design:** Define parametric components with Python code for flexible and efficient designs.
*   **Comprehensive Design Flow:** Includes layout, simulation, optimization, verification (DRC, DFM, LVS), and validation.
*   **Simulation Integration:** Seamlessly integrates with major simulation tools, eliminating the need to redraw components.
*   **Multi-Format Output:** Generates standard CAD file formats like GDSII and OASIS for fabrication. Also includes STL and GERBER files.
*   **Open-Source PDKs:** Access to open-source Process Design Kits (PDKs) for popular foundries, enabling rapid prototyping.
*   **GDSFactory+**: Subscription service with a GUI, PDK access and more!
*   **Growing Community:** Benefit from a vibrant and supportive community, driving continuous innovation.

## Why Choose GDSFactory?

*   **Speed and Efficiency:** Built for performance, especially with large GDS files and complex operations, thanks to the KLayout C++ library.
*   **Open-Source Advantage:** Enjoy the freedom of open-source, with continuous improvements and community contributions.
*   **Extensible and Customizable:** Easily add new components, functionalities, and integrations to fit your specific needs.
*   **End-to-End Design:** Provides a complete workflow from design to validation, saving time and resources.

## Getting Started

GDSFactory offers a gentle learning curve with a friendly community. Get started using these resources:

*   **Documentation:** [https://gdsfactory.github.io/gdsfactory/](https://gdsfactory.github.io/gdsfactory/)
*   **Video Tutorials:** [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/@gdsfactory/playlists)
*   **Community:** Join the discussions and stay informed!
    *   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
    *   [Google Group](https://groups.google.com/g/gdsfactory)
    *   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
    *   [Slack](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)

## PDKs

### Open-Source PDKs (No NDA Required)

*   [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/)
*   [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc)
*   [SkyWater 130nm CMOS PDK](https://gdsfactory.github.io/skywater130/)
*   [VTT PDK](https://github.com/gdsfactory/vtt)
*   [Cornerstone PDK](https://gdsfactory.github.io/cspdk)
*   [Luxtelligence GF PDK](https://github.com/Luxtelligence/lxt_pdk_gf)

### Foundry PDKs (NDA Required)

Access to these PDKs requires a **GDSFactory+** subscription.
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

**GDSFactory+** offers Graphical User Interface for chip design, built on top of GDSFactory and VSCode. It provides you:

*   Foundry PDK access
*   Schematic capture
*   Device and circuit Simulations
*   Design verification (DRC, LVS)
*   Data analytics

## Who is Using GDSFactory?

[Insert logos here, as per original README]

"I've used **GDSFactory** since 2017 for all my chip tapeouts. I love that it is fast, easy to use, and easy to extend. It's the only tool that allows us to have an end-to-end chip design flow (design, verification and validation)."

<div style="text-align: right; margin-right: 10%;">Joaquin Matres - <strong>Google</strong></div>

---

"I've relied on **GDSFactory** for several tapeouts over the years. It's the only tool I've found that gives me the flexibility and scalability I need for a variety of projects."

<div style="text-align: right; margin-right: 10%;">Alec Hammond - <strong>Meta Reality Labs Research</strong></div>

---

"The best photonics layout tool I've used so far and it is leaps and bounds ahead of any commercial alternatives out there. Feels like GDSFactory is freeing photonics."

<div style="text-align: right; margin-right: 10%;">Hasitha Jayatilleka - <strong>LightIC Technologies</strong></div>

---

"As an academic working on large scale silicon photonics at CMOS foundries I've used GDSFactory to go from nothing to full-reticle layouts rapidly (in a few days). I particularly appreciate the full-system approach to photonics, with my layout being connected to circuit simulators which are then connected to device simulators. Moving from legacy tools such as gdspy and phidl to GDSFactory has sped up my workflow at least an order of magnitude."

<div style="text-align: right; margin-right: 10%;">Alex Sludds - <strong>MIT</strong></div>

---

"I use GDSFactory for all of my photonic tape-outs. The Python interface makes it easy to version control individual photonic components as well as entire layouts, while integrating seamlessly with KLayout and most standard photonic simulation tools, both open-source and commercial.

<div style="text-align: right; margin-right: 10%;">Thomas Dorch - <strong>Freedom Photonics</strong></div>

## Contributing

[Reiterate the section about contributions]

## Stargazers

[![Stargazers over time](https://starchart.cc/gdsfactory/gdsfactory.svg)](https://starchart.cc/gdsfactory/gdsfactory)