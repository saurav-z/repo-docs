<div align="center">
<img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Build and Simulate Quantum Circuits with Python

**Cirq is a powerful Python library that empowers you to design, manipulate, and simulate quantum circuits, making quantum computing accessible and easy to explore.**  ([Original Repo](https://github.com/quantumlib/Cirq))

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

## Key Features

*   **Flexible Gate Definitions:** Define custom quantum gates to match your specific needs.
*   **Parameterized Circuits:**  Build circuits with symbolic variables for greater flexibility.
*   **Circuit Transformation and Optimization:**  Simplify and optimize your circuits for better performance.
*   **Hardware Device Modeling:** Model the behavior of real quantum hardware.
*   **Noise Modeling:** Simulate the effects of noise in quantum circuits.
*   **Built-in Simulators:**  Utilize multiple simulators for exploring quantum circuits.
*   **qsim Integration:** Leverage high-performance simulation with integration with [qsim](https://github.com/quantumlib/qsim).
*   **NumPy and SciPy Interoperability:** Seamlessly integrate with popular scientific computing libraries.
*   **Cross-Platform Compatibility:** Works on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq requires Python 3.11 or higher.  Detailed installation instructions are available in the [Install](https://quantumai.google/cirq/start/install) section of the Cirq documentation.

## Quick Start – “Hello Qubit” Example

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

This will output something similar to:

```text
Circuit:
(0, 0): ───X^0.5───M('m')───
Results:
m=11000111111011001000
```

## Documentation

*   **[Cirq Home Page](https://quantumai.google/cirq)**
    *   **[Video tutorials]**
    *   **[Jupyter notebook-based tutorials]**
    *   **[Text-based tutorials]**
        *   [installation]
        *   [basics]
        *   [Build]
        *   [Simulate]
*   **Reference Documentation**
    *   [current stable release]
    *   [pre-release]
*   **Examples**
    *   [examples subdirectory](./examples/)
    *   [Experiments page](https://quantumai.google/cirq/experiments/)
*   **Change log**
    *   [Cirq releases](https://github.com/quantumlib/cirq/releases)

## Integrations

Cirq integrates with several other open-source software projects from Google Quantum AI to help you develop quantum programs for various applications.

<div align="center">

| Your interests                                  | Software to explore  |
|-------------------------------------------------|----------------------|
| Quantum algorithms?<br>Fault-tolerant quantum computing (FTQC)? | [Qualtran] |
| Large circuits and/or a lot of simulations?     | [qsim] |
| Circuits with thousands of qubits and millions of Clifford operations? | [Stim] |
| Quantum error correction (QEC)?                 | [Stim] |
| Chemistry and/or material science?              | [OpenFermion]<br>[OpenFermion-FQE]<br>[OpenFermion-PySCF]<br>[OpenFermion-Psi4] |
| Quantum machine learning (QML)?                 | [TensorFlow Quantum] |
| Real experiments using Cirq?                    | [ReCirq] |

</div>

## Community

Cirq has a thriving community of over 200 contributors!

*   [Contributions](https://github.com/quantumlib/Cirq/graphs/contributors)
*   [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Stay Updated

*   **Announcements:** Subscribe to the low-volume mailing list [`cirq-announce`] for releases and major announcements.
*   **Releases Only:**
    *   GitHub notifications: Configure [repository notifications] for Cirq.
    *   Atom/RSS: Subscribe to the GitHub [Cirq releases Atom feed].
    *   PyPI releases RSS: Subscribe to the [PyPI releases RSS feed].

### Get Involved

*   **Questions and Discussions:** Ask questions and discuss Cirq on the [Quantum Computing Stack Exchange] using the `cirq` tag.
*   **Community Meetings:** Join _Cirq Cynq_, our biweekly virtual meeting. Sign up to [_cirq-dev_] for an invite.
*   **Contributions:**
    *   Report issues: [Open an issue on GitHub]!
    *   Contribute code: Review the [list of good first issues], read our [contribution guidelines], and submit [pull requests]!

## Citing Cirq

When publishing articles, please cite the Cirq version you used.  Download bibliographic records for the latest release:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

For all releases and citation formats, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For questions or concerns, contact quantum-oss-maintainers@google.com.

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
Key improvements and SEO considerations:

*   **Clear Title and Hook:**  The title and the one-sentence hook immediately tell the user what Cirq is and why they should care. The hook includes the keywords "quantum circuits," "Python," and "simulation."
*   **Keyword Optimization:**  Keywords like "quantum circuits," "quantum computing," "simulation," and "Python library" are used throughout the README.
*   **Structured Headings:**  Uses clear, concise headings and subheadings for readability and SEO.
*   **Bulleted Key Features:**  Provides a quick overview of Cirq's capabilities, highlighting the most important features.
*   **Clear Installation and Quick Start:**  Provides straightforward instructions for getting started.
*   **Comprehensive Documentation Links:**  Directs users to various documentation resources, improving findability.
*   **Community and Contribution Sections:**  Encourages community engagement and contributions, which can lead to increased visibility.
*   **Citation Information:** Includes clear instructions on how to cite the library, a best practice.
*   **Concise and Readable:** The text is well-formatted and easy to understand.
*   **Links:**  Includes important links throughout the README, like the links to specific documentation sections, community, and integration partners.

This revised README is significantly more informative, user-friendly, and SEO-optimized. It should help attract more users to the Cirq project.