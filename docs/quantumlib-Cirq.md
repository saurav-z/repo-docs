<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Build, Manipulate, and Run Quantum Circuits

**Cirq** is a powerful Python library for building, manipulating, and running quantum circuits on quantum computers and simulators, empowering researchers and developers to explore the exciting world of quantum computing. ([View on GitHub](https://github.com/quantumlib/Cirq))

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

## Key Features

*   **Flexible Gate Definitions:** Create custom quantum gates to model your specific needs.
*   **Parameterized Circuits:** Utilize symbolic variables for flexible circuit design.
*   **Circuit Transformation & Optimization:** Efficiently compile and optimize your circuits.
*   **Hardware Device Modeling:** Simulate and optimize for real-world hardware constraints.
*   **Noise Modeling:** Incorporate noise to simulate the behavior of real quantum devices.
*   **Built-in Simulators:** Access multiple built-in quantum circuit simulators.
*   **High-Performance Simulation with qsim:** Integrate with qsim for enhanced simulation capabilities.
*   **NumPy and SciPy Integration:** Seamlessly integrate with popular scientific computing libraries.
*   **Cross-Platform Compatibility:** Run Cirq on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq supports Python 3.11 and later.  For detailed installation instructions, consult the [Cirq documentation](https://quantumai.google/cirq/start/install).

## Quick Start - "Hello Qubit" Example

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

## Documentation

The primary documentation is available on the [Cirq home page](https://quantumai.google/cirq).

### Tutorials

*   [Video Tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4)
*   [Jupyter Notebook Tutorials](https://colab.research.google.com/github/quantumlib/Cirq)
*   [Text-Based Tutorials](https://quantumai.google/cirq)

### Reference Documentation

*   [Current Stable Release](https://quantumai.google/reference/python/cirq/all_symbols)
*   [Pre-Release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)

### Examples

*   [Examples Directory](./examples/)
*   [Experiments Page](https://quantumai.google/cirq/experiments/)

### Change Log

*   [Cirq Releases](https://github.com/quantumlib/Cirq/releases)

## Integrations

Cirq seamlessly integrates with other Google Quantum AI tools:

| Your interests                                  | Software to explore  |
|-------------------------------------------------|----------------------|
| Quantum algorithms?<br>Fault-tolerant quantum computing (FTQC)? | [Qualtran] |
| Large circuits and/or a lot of simulations?     | [qsim] |
| Circuits with thousands of qubits and millions of Clifford operations? | [Stim] |
| Quantum error correction (QEC)?                 | [Stim] |
| Chemistry and/or material science?              | [OpenFermion]<br>[OpenFermion-FQE]<br>[OpenFermion-PySCF]<br>[OpenFermion-Psi4] |
| Quantum machine learning (QML)?                 | [TensorFlow Quantum] |
| Real experiments using Cirq?                    | [ReCirq] |

[Qualtran]: https://github.com/quantumlib/qualtran
[qsim]: https://github.com/quantumlib/qsim
[Stim]: https://github.com/quantumlib/stim
[OpenFermion]: https://github.com/quantumlib/openfermion
[OpenFermion-FQE]: https://github.com/quantumlib/OpenFermion-FQE
[OpenFermion-PySCF]: https://github.com/quantumlib/OpenFermion-PySCF
[OpenFermion-Psi4]: https://github.com/quantumlib/OpenFermion-Psi4
[TensorFlow Quantum]: https://github.com/tensorflow/quantum
[ReCirq]: https://github.com/quantumlib/ReCirq

## Community

Cirq thrives on a vibrant and active community.

*   [Contributors](https://github.com/quantumlib/Cirq/graphs/contributors)
*   [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Announcements

Stay updated on Cirq developments:

*   [`cirq-announce` Mailing List](https://groups.google.com/forum/#!forum/cirq-announce) (low-volume)
*   [GitHub Repository Notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications)
*   [Cirq Releases Atom Feed](https://github.com/quantumlib/Cirq/releases.atom)
*   [PyPI Releases RSS Feed](https://pypi.org/rss/project/cirq/releases.xml)

### Questions and Discussions

*   [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) (tag: `cirq`)
*   _Cirq Cynq_ - Biweekly virtual meeting (join [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev) for invitations)

### Contributions

*   [Open an Issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose)
*   [Good First Issues](https://github.com/quantumlib/Cirq/contribute)
*   [Contribution Guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md)
*   [Pull Requests](https://help.github.com/articles/about-pull-requests)

## Citing Cirq

Cite the Cirq version used in your work using the following links:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

For all releases, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any questions or concerns, please contact: quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product.  This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>
```
Key changes and improvements:

*   **SEO-Optimized Title:**  Added a clear, keyword-rich title: "Cirq: Build, Manipulate, and Run Quantum Circuits".
*   **One-Sentence Hook:** Added a concise and engaging introductory sentence to grab the reader's attention.
*   **Clear Headings and Structure:**  Organized the information with clear, descriptive headings for better readability and navigation.
*   **Bulleted Key Features:** Presented the core functionalities in an easy-to-scan bulleted list.
*   **Enhanced Documentation Links:**  Improved the descriptions and links for documentation sections.
*   **Concise and Focused Content:**  Trimmed and rephrased text for better clarity and impact.
*   **Complete and Accurate:** Retained all the essential information from the original README.
*   **Improved Citations Section** Added a direct link to the Zenodo.