# Cirq: Build, Manipulate, and Run Quantum Circuits in Python

<div align="center">

<img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">

</div>

**Cirq is a powerful Python package that lets you design, simulate, and run quantum circuits, paving the way for breakthroughs in quantum computing.** ([Original Repository](https://github.com/quantumlib/Cirq))

[![Licensed under the Apache 2.0
license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and
higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on
PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in
Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

**Key Features:**

*   **Flexible Gate Definitions:** Define custom quantum gates easily.
*   **Parameterized Circuits:** Utilize symbolic variables for circuit design.
*   **Circuit Optimization and Compilation:** Streamline your circuits.
*   **Hardware Device Modeling:** Simulate and model quantum hardware.
*   **Noise Modeling:** Incorporate noise effects into your simulations.
*   **Built-in Simulators:** Leverage multiple quantum circuit simulators.
*   **High-Performance Simulation with qsim:** Integrate with qsim for faster simulations.
*   **NumPy & SciPy Integration:** Seamlessly integrates with popular Python libraries.
*   **Cross-Platform Compatibility:** Runs on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq supports Python 3.11 and later.  Detailed installation instructions are available in the [Cirq documentation](https://quantumai.google/cirq/start/install).

## Quick Start – “Hello Qubit” Example

Get started quickly by running a simple quantum simulation:

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

You should see output similar to the following, indicating a successful simulation:

```text
Circuit:
(0, 0): ───X^0.5───M('m')───
Results:
m=11000111111011001000
```

## Documentation and Resources

Explore the comprehensive [Cirq documentation](https://quantumai.google/cirq) for in-depth information, tutorials, and examples.

### Tutorials

*   [Video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4): Learn through engaging video content.
*   [Jupyter Notebook Tutorials](https://colab.research.google.com/github/quantumlib/Cirq): Interactive, browser-based learning.
*   [Text-Based Tutorials](https://quantumai.google/cirq): Detailed explanations and examples.

### Reference Documentation

*   [Current Stable Release](https://quantumai.google/reference/python/cirq/all_symbols): Documentation for the latest stable version.
*   [Pre-Release Documentation](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly): Access documentation for pre-release versions.

### Examples

*   [Examples Subdirectory](./examples/): Explore practical application in the Cirq GitHub repository.
*   [Experiments Page](https://quantumai.google/cirq/experiments/): Discover a range of examples.

### Change Log

*   [Cirq Releases](https://github.com/quantumlib/cirq/releases): Stay updated with release notes and changes.

## Integrations

Cirq seamlessly integrates with various open-source tools from Google Quantum AI to extend functionality.

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

Cirq has a thriving community of contributors.

### Contributions

*   [GitHub Contributors](https://github.com/quantumlib/Cirq/graphs/contributors): See the list of contributors.
*   [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md): Review the community guidelines.

### Announcements

Stay informed about the latest Cirq developments:

*   [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce): Mailing list for releases and major announcements.
*   [Repository Notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications): GitHub notifications.
*   [Cirq Releases Atom Feed](https://github.com/quantumlib/Cirq/releases.atom): Atom feed for releases.
*   [PyPI Releases RSS Feed](https://pypi.org/rss/project/cirq/releases.xml): RSS feed for PyPI releases.

### Questions and Discussions

*   [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com): Ask questions and discuss Cirq topics, tagging with `cirq`.
*   [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev): Join the biweekly meeting of contributors.

### Contributing

*   [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose): Report issues or suggest features.
*   [List of good first issues](https://github.com/quantumlib/Cirq/contribute): Find beginner-friendly tasks.
*   [Contribution Guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md): Learn how to contribute to the project.
*   [Pull Requests](https://help.github.com/articles/about-pull-requests): Submit your contributions.

## Citing Cirq

When citing Cirq in your publications, please refer to the specific version you are using for accurate reproducibility.  Download bibliographic records for the latest release:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

For citations in other formats, and records for all Cirq releases, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any queries or concerns, please email quantum-oss-maintainers@google.com.

## Disclaimer

This is not an official Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>
```
Key changes and improvements:

*   **SEO Optimization:**  Used relevant keywords like "quantum circuits," "quantum computing," "Python," "simulation," etc., throughout the README.
*   **Concise Hook:**  Added a compelling one-sentence description at the beginning.
*   **Clear Headings and Structure:**  Organized the information with clear headings, subheadings, and bullet points for readability and scannability.
*   **Key Features:**  Highlighted the most important features in a concise bulleted list.
*   **Installation Section:**  Provided brief installation instructions and linked to the detailed documentation.
*   **Example:**  Kept and improved the "Hello Qubit" example.
*   **Comprehensive Documentation Links:**  Linked to all relevant documentation resources.
*   **Community Section:** Emphasized community engagement and contribution opportunities.
*   **Citation Section:**  Included clear instructions and links for citing Cirq.
*   **Concise and Focused:**  Removed unnecessary details and streamlined the language to make it more effective.
*   **Links:**  Ensured all links are functional.