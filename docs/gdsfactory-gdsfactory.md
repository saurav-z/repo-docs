# GDSFactory: Design, Simulate, and Fabricate Chips with Python

**GDSFactory empowers you to design, simulate, and fabricate cutting-edge chips (Photonics, Analog, Quantum, MEMS) using the power of Python.** ([Original Repo](https://github.com/gdsfactory/gdsfactory))

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![Downloads](https://static.pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)

## Key Features

*   **Python-Based Design:** Define parametric components using Python code, enabling efficient and reproducible designs.
*   **Comprehensive Design Flow:** Provides a full end-to-end flow from design to fabrication.
*   **Multi-Format Output:** Generates industry-standard CAD files (GDSII, OASIS, STL, GERBER) for fabrication.
*   **Built-in Simulation Interfaces:** Seamlessly integrates with leading simulation tools, streamlining your workflow.
*   **Verification & Validation:** Includes DRC, DFM, and LVS capabilities for robust design verification.
*   **Open-Source PDKs:** Access a growing library of open-source Process Design Kits (PDKs) for various foundries.
*   **Extensible and Flexible:** Easily add new components, customize functionality, and adapt to your specific needs.
*   **Fast Performance:** Built on the KLayout C++ library for superior speed, especially with large designs.

## Why Choose GDSFactory?

*   **Accelerate Your Design:** Design chips with Python and generate the CAD files you need.
*   **Open-Source Advantage:** Benefit from a thriving open-source community, continuous improvements, and transparency.
*   **Efficiency and Flexibility:** GDSFactory is designed for efficiency and flexibility, allowing you to streamline your design process.
*   **Integration with Tools:** Easy to use with all major simulation tools, simplifying the workflow.

## Quick Start

Get started in minutes:

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

Access a range of publicly available PDKs:

*   [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/)
*   [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc)
*   [SkyWater 130nm CMOS PDK](https://gdsfactory.github.io/skywater130/)
*   [VTT PDK](https://github.com/gdsfactory/vtt)
*   [Cornerstone PDK](https://gdsfactory.github.io/cspdk)
*   [Luxtelligence GF PDK](https://github.com/Luxtelligence/lxt_pdk_gf)

## Foundry PDKs (NDA Required)

For access to the following PDKs, a **GDSFactory+** subscription is required. Visit [GDSFactory.com](https://gdsfactory.com/) to sign up.

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

**GDSFactory+** offers a Graphical User Interface (GUI) built on top of GDSFactory and VSCode, providing:

*   Foundry PDK access
*   Schematic capture
*   Device and circuit Simulations
*   Design verification (DRC, LVS)
*   Data analytics

## Community

Join our growing community for support and collaboration:

*   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
*   [Google Group](https://groups.google.com/g/gdsfactory)
*   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [Slack community channel](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)