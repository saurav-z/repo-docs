# GDSFactory: Design and Fabricate Chips with Python

**GDSFactory empowers you to design chips (photonics, analog, quantum, MEMS), PCBs, and 3D-printable objects, all using Python.**

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![Downloads](https://static.pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)

[View the original repository on GitHub](https://github.com/gdsfactory/gdsfactory)

## Key Features

*   **Python-Based Design:** Define parametric components using Python code for flexible and repeatable designs.
*   **Versatile Output:** Generate industry-standard CAD files, including GDSII, OASIS, STL, and GERBER.
*   **Simulation Integration:** Seamlessly integrate with leading simulation tools for accurate analysis and verification.
*   **Comprehensive Verification:** Built-in capabilities for Design Rule Checking (DRC), Design for Manufacturing (DFM), and Layout Versus Schematic (LVS).
*   **Automated Validation:** Implement automated chip analysis and data pipelines for post-fabrication evaluation.
*   **Open-Source PDK Support:** Access a growing library of open-source Process Design Kits (PDKs) for various fabrication processes.
*   **GDSFactory+:** Subscription-based access to foundry PDKs, a GUI, and enhanced design and verification tools.

## Design, Simulate, and Fabricate with Ease

GDSFactory provides an end-to-end design flow, streamlining the process from concept to fabrication.

*   **Design:** Define and generate components using Python, test component settings and geometry.
*   **Verify:** Run simulations directly from your layout, conduct component and circuit simulations, and ensure design integrity with DRC and LVS.
*   **Validate:** Define layout and test protocols for automated chip analysis.

Your input is Python code, and your output is fabrication-ready GDSII or OASIS files, accompanied by component settings, netlists, and more.

![cad](https://i.imgur.com/3cUa2GV.png)

## Quick Start

Get started with a simple example:

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

*   **Speed:** Experience rapid performance, thanks to the KLayout C++ library.
*   **Flexibility:** Designed for extensibility, making it easy to tailor your workflow.
*   **Open Source:** Benefit from a free and open-source tool with a thriving community.
*   **Community Driven:** Leverage the collective knowledge and contributions of a growing ecosystem.

## Downloads and Contributors

*   **+2M Downloads**
*   **+81 Contributors**
*   **+25 PDKs available**

![workflow](https://i.imgur.com/KyavbHh.png)

## Open-Source PDKs

These PDKs are publicly available:

*   [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/)
*   [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc)
*   [SkyWater 130nm CMOS PDK](https://gdsfactory.github.io/skywater130/)
*   [VTT PDK](https://github.com/gdsfactory/vtt)
*   [Cornerstone PDK](https://gdsfactory.github.io/cspdk)
*   [Luxtelligence GF PDK](https://github.com/Luxtelligence/lxt_pdk_gf)

## Foundry PDKs (GDSFactory+ Subscription Required)

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

## GDSFactory+ Features

**GDSFactory+** offers a user-friendly Graphical User Interface for chip design built on top of GDSFactory and VSCode. It offers:

*   Foundry PDK access
*   Schematic capture
*   Device and circuit Simulations
*   Design verification (DRC, LVS)
*   Data analytics

## Performance Benchmarks

GDSFactory leverages the KLayout C++ library for superior performance:

| Benchmark      |  gdspy  | GDSFactory | Gain |
| :------------- | :-----: | :--------: | :--: |
| 10k_rectangles | 80.2 ms |  4.87 ms   | 16.5 |
| boolean-offset | 187 μs  |  44.7 μs   | 4.19 |
| bounding_box   | 36.7 ms |   170 μs   | 216  |
| flatten        | 465 μs  |  8.17 μs   | 56.9 |
| read_gds       | 2.68 ms |   94 μs    | 28.5 |

## Getting Started & Resources

*   [See slides](https://docs.google.com/presentation/d/1_ZmUxbaHWo_lQP17dlT1FWX-XD8D9w7-FcuEih48d_0/edit#slide=id.g11711f50935_0_5)
*   [Read docs](https://gdsfactory.github.io/gdsfactory/)
*   [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/@gdsfactory/playlists)
*   See announcements on [GitHub](https://github.com/gdsfactory/gdsfactory/discussions/547), [google-groups](https://groups.google.com/g/gdsfactory) or [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=250169028)
*   [PIC training](https://gdsfactory.github.io/gdsfactory-photonics-training/)
*   Online course [UBCx: Silicon Photonics Design, Fabrication and Data Analysis](https://www.edx.org/learn/engineering/university-of-british-columbia-silicon-photonics-design-fabrication-and-data-ana), where students can use GDSFactory to create a design, have it fabricated, and tested.
*   [Visit website](https://gdsfactory.com)

## Who is Using GDSFactory?

Hundreds of organizations are using GDSFactory, including:

![logos](https://i.imgur.com/VzLNMH1.png)

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

## Contributors

A huge thanks to all the contributors who make this project possible!

We welcome all contributions—whether you're adding new features, improving documentation, or even fixing a small typo. Every contribution helps make GDSFactory better!

Join us and be part of the community. 🚀

![contributors](https://i.imgur.com/0AuMHZE.png)

## Stargazers

[![Stargazers over time](https://starchart.cc/gdsfactory/gdsfactory.svg)](https://starchart.cc/gdsfactory/gdsfactory)

## Community

Join our growing community:

*   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
*   [Google Group](https://groups.google.com/g/gdsfactory)
*   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [Slack community channel](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)