<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Build and Simulate Quantum Circuits with Python

**Cirq is a powerful Python package for designing, manipulating, and simulating quantum circuits, making it easier than ever to explore the world of quantum computing.**

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

[**View the Cirq GitHub Repository**](https://github.com/quantumlib/Cirq)

**Key Features:**

*   **Flexible Gate Definitions:** Create custom quantum gates to match your specific needs.
*   **Parameterized Circuits:** Utilize symbolic variables for dynamic circuit design.
*   **Circuit Transformation & Optimization:** Simplify and optimize your circuits for efficient execution.
*   **Hardware Device Modeling:** Model the characteristics of real-world quantum hardware.
*   **Noise Modeling:** Simulate the effects of noise on your quantum circuits.
*   **Multiple Simulators:** Access built-in simulators for efficient testing and analysis.
*   **High-Performance Simulation:** Integrate with [qsim](https://github.com/quantumlib/qsim) for enhanced performance.
*   **Interoperability:** Seamlessly integrate with [NumPy](https://numpy.org) and [SciPy](https://scipy.org).
*   **Cross-Platform Compatibility:** Works on Linux, macOS, Windows, and [Google Colab](https://colab.google).

## Getting Started

### Installation

Cirq requires Python 3.11 or later.  Detailed installation instructions can be found in the [Install](https://quantumai.google/cirq/start/install) section of the Cirq documentation.

### Quick Start Example

Here's a basic example to get you started:

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

This will output a circuit and simulated results, demonstrating a fundamental quantum operation.

## Documentation & Resources

Explore the comprehensive documentation to learn more about Cirq:

### Tutorials

*   [Video Tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4) - Engaging video guides.
*   [Jupyter Notebook Tutorials](https://colab.research.google.com/github/quantumlib/Cirq) - Interactive tutorials in your browser.
*   [Text-Based Tutorials](https://quantumai.google/cirq) - Detailed guides on the Cirq website.

### Reference Documentation

*   [Current Stable Release Documentation](https://quantumai.google/reference/python/cirq/all_symbols)
*   [Pre-Release Documentation](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)

### Examples

*   [Cirq Examples in GitHub](./examples/) - Code examples for various applications.
*   [Cirq Experiments](https://quantumai.google/cirq/experiments/) - Advanced use cases.

### Change Log

*   [Cirq Releases](https://github.com/quantumlib/Cirq/releases) - Track changes and updates.

## Integrations

Cirq seamlessly integrates with other Google Quantum AI open-source projects:

| Your Interests                             | Software to Explore                                  |
| ------------------------------------------ | ---------------------------------------------------- |
| Quantum algorithms / FTQC                | [Qualtran](https://github.com/quantumlib/qualtran)     |
| Large circuits/Simulations                 | [qsim](https://github.com/quantumlib/qsim)           |
| Thousands of qubits/Clifford operations    | [Stim](https://github.com/quantumlib/stim)           |
| Quantum error correction (QEC)             | [Stim](https://github.com/stim)                      |
| Chemistry/Material Science                 | [OpenFermion](https://github.com/quantumlib/openfermion), [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE), [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF), [OpenFermion-Psi4](https://github.com/quantumlib/OpenFermion-Psi4) |
| Quantum Machine Learning (QML)             | [TensorFlow Quantum](https://github.com/tensorflow/quantum) |
| Real experiments using Cirq               | [ReCirq](https://github.com/quantumlib/ReCirq)        |

## Community

Cirq has a thriving community with over 200 contributors. We encourage collaboration and inclusivity.

*   [Contributions](https://github.com/quantumlib/Cirq/graphs/contributors)
*   [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Stay Updated

*   **Announcements:** Subscribe to the [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce) mailing list.
*   **Releases:**  Follow the [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom) or the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).

### Get Involved

*   **Ask Questions:** Post to the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) with the tag `cirq`.
*   **Discussions:** Join the _Cirq Cynq_ meetings by subscribing to the  [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev) mailing list.
*   **Contribute:**  [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose) for feature requests or bug reports.  Review the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md) and start contributing with [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

When referencing Cirq in your work, please cite the specific version you use. You can download the bibliographic record for the latest stable release:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

For citations and records in other formats, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any other questions or concerns, contact us at quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>
```
Key improvements and explanations:

*   **SEO-Optimized Title and Description:**  The title includes keywords like "quantum circuits," "Python," and "simulation" to improve search visibility. The first sentence is a concise and engaging hook.
*   **Clear Headings:**  Uses `##` for clear sectioning (Features, Installation, etc.) for readability and SEO.
*   **Bulleted Key Features:**  Uses bullet points to highlight the core capabilities of Cirq.
*   **Detailed Installation Section:**  Provides clear installation information, including the Python version requirement and links to the detailed documentation.
*   **Comprehensive Documentation Links:** Provides direct links to key documentation sections, including tutorials, reference documentation, and examples, making it easy for users to find information.
*   **Integration Section:**  Organized the integrations into a clear table for quick reference.
*   **Community and Contribution Sections:**  Emphasizes community involvement and how to contribute.
*   **Citing Cirq:** Explains how to cite Cirq and provides direct links to download citation records.
*   **Concise Contact and Disclaimer:**  Keeps the contact and disclaimer information brief.
*   **Clean formatting:** Used bold and italics appropriately for emphasis.
*   **Removed unnecessary elements:** Removed the original README's links section as links were already available in other sections.
*   **Improved Readability:**  Uses concise language and formatting for better readability.

This improved version is much more user-friendly, SEO-optimized, and provides a more compelling overview of the Cirq library.