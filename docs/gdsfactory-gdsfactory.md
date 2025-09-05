# GDSFactory: Design Chips with Python - Revolutionizing Hardware Design

**GDSFactory** is a powerful Python library for designing chips (Photonics, Analog, Quantum, MEMS), PCBs, and 3D-printable objects, making hardware design accessible, intuitive, and fun. Build the future with the power of Python and open-source!  [Explore the original repo](https://github.com/gdsfactory/gdsfactory)

![GDSFactory CAD example](https://i.imgur.com/3cUa2GV.png)

## Key Features

*   **Python-Driven Design:** Define parametric components with Python code for flexible and reusable designs.
*   **Versatile Output:** Generate industry-standard GDSII, OASIS, STL, and GERBER files for fabrication.
*   **Seamless Simulation Integration:** Direct integration with leading simulation tools to streamline your workflow.
*   **Comprehensive Verification:** Built-in DRC (Design Rule Checking), DFM (Design for Manufacturing), and LVS (Layout Versus Schematic) capabilities to ensure design accuracy.
*   **Automated Validation:** Enables automated chip analysis and data pipelines for post-fabrication performance monitoring.
*   **Open-Source Advantage**: Benefit from a thriving community, continuous contributions, and transparent development, just like top machine learning libraries.
*   **Free and Open Source**: No licensing fees - modify and extend it as you need.
*   **Fast & Extensible:** Designed for efficiency and flexibility, offering significant performance gains compared to other tools (see benchmark below).

## Getting Started

**Quickly design and export layouts using Python:**

1.  **Install:** `pip install gdsfactory_install` and `gfi install`
2.  **Create & Show:**

    ```python
    import gdsfactory as gf

    c = gf.Component()
    r = gf.components.rectangle(size=(10, 10), layer=(1, 0))
    rect = c.add_ref(r)
    t1 = gf.components.text("Hello", size=10, layer=(2, 0))
    text1 = c.add_ref(t1)
    text1.xmin = rect.xmax + 5
    c.show()
    ```

## Performance Benchmarks

GDSFactory is optimized for speed, leveraging the KLayout C++ library for fast processing of GDS objects.

| Benchmark      | gdspy  | GDSFactory | Gain |
| :------------- | :-----: | :--------: | :--: |
| 10k_rectangles | 80.2 ms |  4.87 ms   | 16.5 |
| boolean-offset | 187 μs  |  44.7 μs   | 4.19 |
| bounding_box   | 36.7 ms |   170 μs   | 216  |
| flatten        | 465 μs  |  8.17 μs   | 56.9 |
| read_gds       | 2.68 ms |   94 μs    | 28.5 |

## Who Uses GDSFactory?

GDSFactory is trusted by a diverse range of organizations worldwide, including:

![logos](https://i.imgur.com/VzLNMH1.png)

>   "I've used **GDSFactory** since 2017 for all my chip tapeouts. I love that it is fast, easy to use, and easy to extend. It's the only tool that allows us to have an end-to-end chip design flow (design, verification and validation)." - Joaquin Matres, Google

>   "I've relied on **GDSFactory** for several tapeouts over the years. It's the only tool I've found that gives me the flexibility and scalability I need for a variety of projects." - Alec Hammond, Meta Reality Labs Research

## Open-Source and Foundry PDKs

**Open-Source PDKs (No NDA Required):**

*   [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/)
*   [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc)
*   [SkyWater 130nm CMOS PDK](https://gdsfactory.github.io/skywater130/)
*   [VTT PDK](https://github.com/gdsfactory/vtt)
*   [Cornerstone PDK](https://gdsfactory.github.io/cspdk)
*   [Luxtelligence GF PDK](https://github.com/Luxtelligence/lxt_pdk_gf)

**Foundry PDKs (GDSFactory+ Subscription Required):**

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

## GDSFactory+: Advanced Features

**GDSFactory+** offers an advanced Graphical User Interface for chip design, built on top of GDSFactory and VSCode. It provides:

*   Foundry PDK access
*   Schematic capture
*   Device and circuit Simulations
*   Design verification (DRC, LVS)
*   Data analytics

Visit [GDSFactory.com](https://gdsfactory.com/) to sign up for GDSFactory+.

## Resources

*   [Documentation](https://gdsfactory.github.io/gdsfactory/)
*   [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/@gdsfactory/playlists)
*   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions/547)
*   [Google Group](https://groups.google.com/g/gdsfactory)
*   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=250169028)
*   [PIC training](https://gdsfactory.github.io/gdsfactory-photonics-training/)
*   Online course [UBCx: Silicon Photonics Design, Fabrication and Data Analysis](https://www.edx.org/learn/engineering/university-of-british-columbia-silicon-photonics-design-fabrication-and-data-ana)
*   [Visit website](https://gdsfactory.com)

## Join the Community

Be part of the GDSFactory community!

*   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
*   [Google Group](https://groups.google.com/g/gdsfactory)
*   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [Slack community channel](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)

## Contributors

A big thank you to all contributors!  Contributions are welcome!

![contributors](https://i.imgur.com/0AuMHZE.png)

## Star History

[![Star History](https://api.star-history.com/svg?repos=gdsfactory/gdsfactory&type=Date)](https://star-history.com/#gdsfactory/gdsfactory)