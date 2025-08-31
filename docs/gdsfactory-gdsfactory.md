# GDSFactory: Design, Simulate, and Fabricate Chips with Python

**GDSFactory empowers engineers and researchers to design, simulate, and fabricate advanced chips, photonics, and other micro-fabricated devices using Python.**  [Check out the original repo](https://github.com/gdsfactory/gdsfactory).

GDSFactory is a powerful, open-source Python library for designing and fabricating chips (Photonics, Analog, Quantum, MEMS), PCBs, and 3D-printable objects. It streamlines the entire hardware design process, from initial design to fabrication-ready files.

## Key Features:

*   **Parametric Design with Python:** Define and create complex components using intuitive Python code.
*   **Versatile Output Formats:** Generate industry-standard CAD files (GDS, OASIS, STL, GERBER) for fabrication.
*   **Integrated Simulation:** Seamlessly integrate with leading simulation tools to model and analyze your designs.
*   **Built-in Verification:** Comprehensive DRC (Design Rule Checking), DFM (Design for Manufacturing), and LVS (Layout Versus Schematic) capabilities ensure design accuracy and manufacturability.
*   **Automated Validation:** Establish automated chip analysis and data pipelines for efficient post-fabrication evaluation.
*   **Open-Source PDKs:** Access a range of open-source Process Design Kits (PDKs) for various fabrication processes, enabling rapid prototyping and experimentation.
*   **Rapid Performance:** Built on a highly efficient C++ backend (KLayout) for exceptional speed, especially with large designs.
*   **Thriving Community:** Benefit from a vibrant and active community, providing support, resources, and continuous improvements.

## Why Choose GDSFactory?

*   **Speed:** GDSFactory leverages the KLayout C++ library for blazingly fast performance, especially when handling large GDS files and complex operations.
*   **Extensibility:** Designed to be easily extended, allowing you to add custom components, features, and integrations.
*   **Open-Source Advantage:** Benefit from the collaborative power of open-source, with contributions from a growing community, continuous improvements, and transparency.
*   **Comprehensive Design Flow:** Offers a complete, end-to-end workflow for chip design, from design to validation.
*   **Large Community:** The EDA tool with one of the most active communities.

## Performance Benchmarks (Compared to gdspy)

| Benchmark          | gdspy  | GDSFactory | Gain |
| :----------------- | :-----: | :--------: | :--: |
| 10k_rectangles     | 80.2 ms |  4.87 ms   | 16.5 |
| boolean-offset     | 187 μs  |  44.7 μs   | 4.19 |
| bounding_box       | 36.7 ms |   170 μs   | 216  |
| flatten            | 465 μs  |  8.17 μs   | 56.9 |
| read_gds           | 2.68 ms |   94 μs    | 28.5 |

## Get Started:

Install GDSFactory:
```bash
pip install gdsfactory_install
gfi install
```

Then explore these resources:

*   [Documentation](https://gdsfactory.github.io/gdsfactory/)
*   [YouTube Tutorials](https://www.youtube.com/@gdsfactory/playlists)
*   [GitHub Discussions](https://github.com/gdsfactory/gdsfactory/discussions)
*   [Google Group](https://groups.google.com/g/gdsfactory)
*   [LinkedIn](https://www.linkedin.com/company/gdsfactory)
*   [Slack community channel](https://join.slack.com/t/gdsfactory-community/shared_invite/zt-3aoygv7cg-r5BH6yvL4YlHfY8~UXp0Wg)
*   [Open in GitHub Codespaces](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=250169028)
*   [PIC training](https://gdsfactory.github.io/gdsfactory-photonics-training/)

## Who's Using GDSFactory?

Join the ranks of hundreds of organizations around the world utilizing GDSFactory: (See the original repo for logos)

## Contributors

A huge thanks to all the contributors who make this project possible! (See the original repo for the contributor image)

We welcome all contributions—whether you're adding new features, improving documentation, or even fixing a small typo. Every contribution helps make GDSFactory better!
Join us and be part of the community.