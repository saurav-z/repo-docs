<div align="center">
  <a href="https://github.com/quantumlib/Cirq">
    <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
  </a>
</div>

# Cirq: Build and Simulate Quantum Circuits with Python

**Cirq is a powerful Python library that empowers you to write, manipulate, and run quantum circuits, providing a bridge between theory and practical quantum computing.**

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

[**Key Features**](#features) | [**Installation**](#installation) | [**Quick Start**](#quick-start--hello-qubit-example) | [**Documentation**](#cirq-documentation) | [**Integrations**](#integrations) | [**Community**](#community) | [**Citing Cirq**](#citing-cirq) | [**Contact**](#contact)

## Key Features

Cirq offers a comprehensive suite of tools for quantum circuit development and simulation, specifically designed for **noisy intermediate-scale quantum (NISQ)** computers:

*   **Flexible Gate Definitions:** Define custom quantum gates to match your specific needs.
*   **Parameterized Circuits:** Build circuits with symbolic variables for dynamic control.
*   **Circuit Transformation & Optimization:**  Optimize and manipulate your circuits for better performance.
*   **Hardware Device Modeling:** Simulate circuits on specific quantum hardware devices.
*   **Noise Modeling:**  Simulate the effects of noise in real quantum systems.
*   **Multiple Built-in Simulators:** Use various simulation methods.
*   **High-Performance Simulation with qsim:** Benefit from integration with [qsim](https://github.com/quantumlib/qsim) for fast simulation.
*   **Seamless Interoperability:** Works well with [NumPy](https://numpy.org) and [SciPy](https://scipy.org) for data analysis.
*   **Cross-Platform Compatibility:** Use Cirq on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq requires Python 3.11 or later.  Detailed installation instructions, including how to install Cirq on your local machine or in Google Colab, are available in the [Cirq documentation](https://quantumai.google/cirq/start/install).

## Quick Start – “Hello Qubit” Example

Get started quickly with this simple example:

```python
import cirq

# Pick a qubit.
qubit = cirq.GridQubit(0, 0)

# Create a circuit.
circuit = cirq.Circuit(
    cirq.X(qubit)**0.5,  # Square root of NOT.
    cirq.measure(qubit, key='m')  # Measurement.
)
print("Circuit:")
print(circuit)

# Simulate the circuit several times.
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=20)
print("Results:")
print(result)
```

This code will output the circuit and the results of a simulation, demonstrating how easy it is to get started with Cirq.

## Cirq Documentation

Comprehensive documentation is available on the [Cirq home page on the Quantum AI website](https://quantumai.google/cirq).

### Tutorials

*   **Video Tutorials:** Learn through engaging [video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4).
*   **Jupyter Notebook Tutorials:** Interactive tutorials using [Jupyter notebooks](https://colab.research.google.com/github/quantumlib/Cirq).
*   **Text-Based Tutorials:** Step-by-step guides on the Cirq [home page](https://quantumai.google/cirq), with tutorials on circuit building and simulation.

### Reference Documentation

*   Docs for the [current stable release](https://quantumai.google/reference/python/cirq/all_symbols)
*   Docs for the [pre-release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)

### Examples

*   Explore the [examples subdirectory](./examples/) of the Cirq GitHub repo for various applications.
*   Find additional examples on the [Experiments page](https://quantumai.google/cirq/experiments/) in the Cirq documentation.

### Change Log

*   View the [Cirq releases](https://github.com/quantumlib/cirq/releases) page on GitHub to see what has changed in each release.

## Integrations

Cirq seamlessly integrates with other powerful tools from Google Quantum AI:

<div align="center">

| Your Interests                                       | Software to Explore                                          |
|------------------------------------------------------|--------------------------------------------------------------|
| Quantum Algorithms, Fault-Tolerant Quantum Computing | [Qualtran](https://github.com/quantumlib/qualtran)          |
| Large Circuits and Simulations                       | [qsim](https://github.com/quantumlib/qsim)                   |
| Circuits with Millions of Operations                 | [Stim](https://github.com/stim)                               |
| Quantum Error Correction (QEC)                       | [Stim](https://github.com/stim)                               |
| Chemistry and Material Science                       | [OpenFermion](https://github.com/quantumlib/openfermion)<br>[OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)<br>[OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF)<br>[OpenFermion-Psi4](https://github.com/quantumlib/OpenFermion-Psi4) |
| Quantum Machine Learning (QML)                       | [TensorFlow Quantum](https://github.com/tensorflow/quantum)     |
| Real Experiments Using Cirq                          | [ReCirq](https://github.com/quantumlib/ReCirq)                |

</div>

## Community

Cirq thrives with a vibrant community of contributors.

<a href="https://github.com/quantumlib/Cirq/graphs/contributors"><img
width="150em" alt="Total number of contributors to Cirq"
src="https://img.shields.io/github/contributors/quantumlib/cirq?label=Contributors&logo=github&color=ccc&style=flat-square"/></a>

Learn more about contributing to Cirq:

*   [Contributions](https://github.com/quantumlib/Cirq/graphs/contributors)
*   [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Announcements

Stay informed about Cirq developments:

*   **Mailing List:** Sign up for the low-volume [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce) mailing list.
*   **GitHub Notifications:**  Configure [repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications) for Cirq.
*   **Atom/RSS Feeds:** Subscribe to the GitHub [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom) or the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).

Cirq releases happen approximately every quarter.

### Questions and Discussions

*   **Quantum Computing Stack Exchange:** Ask questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) and tag them with [`cirq`](https://quantumcomputing.stackexchange.com/questions/tagged/cirq).
*   **Cirq Cynq:** Join our biweekly virtual meeting for discussions and collaboration. Sign up for [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev) to get a meeting invitation.

### Contributions

*   **Report Issues:** [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose) for feature requests or bug reports.
*   **Develop Code:**  Explore the [list of good first issues](https://github.com/quantumlib/Cirq/contribute), review the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md), and submit [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

When referencing Cirq in your research, please cite the specific version you are using.

<div align="center">

[![Download BibTeX bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

For more citation formats and all Cirq releases, see the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For inquiries, contact quantum-oss-maintainers@google.com.

## Disclaimer

This is not an official Google product and is not covered by the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>
```
Key improvements and SEO enhancements:

*   **Concise Title:** The original title was replaced with a clear and keyword-rich title.
*   **One-Sentence Hook:** A compelling introduction that highlights the core purpose and value of Cirq.
*   **Keyword Optimization:** Incorporated relevant keywords such as "quantum circuits," "Python," "quantum computing," and "simulation" throughout the document.
*   **Clear Headings and Structure:** Improved the organization with clear headings and subheadings for easy navigation and readability.
*   **Bullet Points:**  Used bullet points to enhance readability and highlight key features.
*   **External Links:** Provided external links to relevant resources (documentation, tutorials, etc.).
*   **Call to Action:** Encouraged users to explore the documentation and community resources.
*   **SEO-Friendly Formatting:** Used Markdown formatting effectively for headings, lists, and emphasis.
*   **Zenodo Citation:** Emphasized the importance of citing Cirq and made it easy to download citation information.
*   **Contact Information:** Included contact information for support.