# GDSFactory: Design Chips (Photonics, Analog, Quantum) with Python

**GDSFactory** is a powerful Python library that empowers engineers and researchers to design and fabricate chips for photonics, analog, quantum computing, MEMS, and PCBs, making hardware design accessible and efficient.  [Visit the original repository](https://github.com/gdsfactory/gdsfactory) for more information.

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![Downloads](https://static.pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)

## Key Features

*   **Intuitive Design:** Design parametric cells using Python code.
*   **Multi-Platform Compatibility:** Generates industry-standard CAD files including GDSII, OASIS, STL, and GERBER.
*   **Comprehensive Verification:** Includes built-in DRC (Design Rule Checking), DFM (Design for Manufacturing), and LVS (Layout Versus Schematic) capabilities.
*   **Seamless Simulation Integration:** Supports direct integration with major simulation tools.
*   **Automated Validation:** Enables automated chip analysis and data pipelines for post-fabrication testing.
*   **Open-Source PDKs:** Access and utilize open-source Process Design Kits (PDKs) for various technologies.
*   **GDSFactory+ (Paid):**  Offers a GUI, schematic capture, simulation, verification, and analytics.

## Why Choose GDSFactory?

GDSFactory offers a streamlined, efficient, and flexible approach to chip design, making it a leading choice for both researchers and industry professionals. GDSFactory is fast thanks to the KLayout C++ library for manipulating GDS objects.

| Benchmark      |  gdspy  | GDSFactory | Gain |
| :------------- | :-----: | :--------: | :--: |
| 10k_rectangles | 80.2 ms |  4.87 ms   | 16.5 |
| boolean-offset | 187 μs  |  44.7 μs   | 4.19 |
| bounding_box   | 36.7 ms |   170 μs   | 216  |
| flatten        | 465 μs  |  8.17 μs   | 56.9 |
| read_gds       | 2.68 ms |   94 μs    | 28.5 |

## Getting Started

Install the library:

```bash
pip install gdsfactory_install
gfi install
```

Then, jump right into chip design with Python:

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

## PDKs

### Open-Source PDKs (No NDA Required)

-   [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/)
-   [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc)
-   [SkyWater 130nm CMOS PDK](https://gdsfactory.github.io/skywater130/)
-   [VTT PDK](https://github.com/gdsfactory/vtt)
-   [Cornerstone PDK](https://gdsfactory.github.io/cspdk)
-   [Luxtelligence GF PDK](https://github.com/Luxtelligence/lxt_pdk_gf)

### Foundry PDKs (NDA Required, GDSFactory+ subscription)

-   AIM Photonics
-   AMF Photonics
-   CompoundTek Photonics
-   Fraunhofer HHI Photonics
-   Smart Photonics
-   Tower Semiconductor PH18
-   Tower PH18DA by OpenLight
-   III-V Labs
-   LioniX
-   Ligentec
-   Lightium
-   Quantum Computing Inc. (QCI)

## Community

*   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
*   [Google Group](https://groups.google.com/g/gdsfactory)
*   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [Slack community channel](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)