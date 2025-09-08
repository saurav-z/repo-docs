<!-- Improved README.md for GDSFactory -->

# GDSFactory: Design & Fabricate Chips (Photonics, Analog, Quantum, MEMS) with Python

**GDSFactory empowers engineers and researchers to design, simulate, and fabricate chips with an intuitive Python interface, unlocking the future of hardware design.**

[View the original repository on GitHub](https://github.com/gdsfactory/gdsfactory)

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![Downloads](https://static.pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)

## Key Features

*   **Intuitive Python Interface:** Design components and layouts using Python code.
*   **Multi-Disciplinary Support:**  Design chips for photonics, analog circuits, quantum computing, and MEMS.
*   **Automated Design Flow:** Streamlined design, simulation, verification (DRC, LVS), and validation.
*   **Open-Source & Extensible:** Benefit from a growing community and easily add custom components.
*   **CAD File Generation:** Create industry-standard GDSII, OASIS, STL, and GERBER files.
*   **Simulation Integration:** Seamlessly integrates with popular simulation tools.
*   **Foundry-Ready:** Support for various PDKs (Process Design Kits), with both open and NDA-required options.
*   **Fast Performance:** Leveraging the KLayout C++ library for efficient handling of large designs.

## Core Functionality

GDSFactory offers a comprehensive end-to-end design flow:

*   **Design (Layout, Simulation, Optimization):** Define parametric cell functions in Python to generate components. Test component settings, ports, and geometry to avoid unwanted regressions, and capture design intent in a schematic.
*   **Verify (DRC, DFM, LVS):** Run simulations directly from the layout using our simulation interfaces, removing the need to redraw your components in simulation tools. Conduct component and circuit simulations, study design for manufacturing. Ensure complex layouts match their design intent through Layout Versus Schematic verification (LVS) and are DRC clean.
*   **Validate:** Define layout and test protocols simultaneously for automated chip analysis post-fabrication. This allows you to extract essential component parameters, and build data pipelines from raw data to structured data to monitor chip performance.

## Quick Start

Get started with GDSFactory in minutes:

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

Access a range of open-source PDKs:

*   [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/)
*   [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc)
*   [SkyWater 130nm CMOS PDK](https://gdsfactory.github.io/skywater130/)
*   [VTT PDK](https://github.com/gdsfactory/vtt)
*   [Cornerstone PDK](https://gdsfactory.github.io/cspdk)
*   [Luxtelligence GF PDK](https://github.com/Luxtelligence/lxt_pdk_gf)

## Foundry PDKs (NDA Required)

GDSFactory+ subscribers have access to advanced PDKs. Sign up at [GDSFactory.com](https://gdsfactory.com/).
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

**GDSFactory+** offers a user-friendly Graphical User Interface (GUI) on top of GDSFactory, powered by VSCode, for:

*   Foundry PDK access
*   Schematic capture
*   Device and circuit simulations
*   Design verification (DRC, LVS)
*   Data analytics

## Why Choose GDSFactory?

*   **Speed and Efficiency:**  Benefit from fast performance optimized using the KLayout C++ library.
*   **Open and Collaborative:** Participate in a vibrant open-source community.
*   **Flexibility and Control:**  Customize your design flow with Python scripting.
*   **End-to-End Solution:**  From design to fabrication, GDSFactory provides a complete workflow.

## Benchmarks

GDSFactory delivers significant performance gains compared to other tools:

| Benchmark      |  gdspy  | GDSFactory | Gain |
| :------------- | :-----: | :--------: | :--: |
| 10k_rectangles | 80.2 ms |  4.87 ms   | 16.5 |
| boolean-offset | 187 μs  |  44.7 μs   | 4.19 |
| bounding_box   | 36.7 ms |   170 μs   | 216  |
| flatten        | 465 μs  |  8.17 μs   | 56.9 |
| read_gds       | 2.68 ms |   94 μs    | 28.5 |

## Community

Join our growing community and connect with other users and developers:

*   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
*   [Google Group](https://groups.google.com/g/gdsfactory)
*   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [Slack community channel](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)

## Resources

*   [Documentation](https://gdsfactory.github.io/gdsfactory/)
*   [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/@gdsfactory/playlists)
*   [PIC training](https://gdsfactory.github.io/gdsfactory-photonics-training/)
*   Online course [UBCx: Silicon Photonics Design, Fabrication and Data Analysis](https://www.edx.org/learn/engineering/university-of-british-columbia-silicon-photonics-design-fabrication-and-data-ana)

## Who's Using GDSFactory?

Join the hundreds of organizations using GDSFactory for their chip design needs:

![logos](https://i.imgur.com/VzLNMH1.png)

(Include quotes from users here - consider formatting them as blockquotes.)

## Contributors

We are grateful for all contributions to this project.  Join us in making GDSFactory even better!

![contributors](https://i.imgur.com/0AuMHZE.png)

## Stargazers

[![Stargazers over time](https://starchart.cc/gdsfactory/gdsfactory.svg)](https://starchart.cc/gdsfactory/gdsfactory)