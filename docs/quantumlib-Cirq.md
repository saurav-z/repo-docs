<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Build, Manipulate, and Run Quantum Circuits with Python

**Cirq** is a powerful Python package for designing and simulating quantum circuits, empowering researchers and developers to explore the cutting-edge world of quantum computing. ([See the original repo](https://github.com/quantumlib/Cirq))

[![Licensed under the Apache 2.0
license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and
higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on
PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in
Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

## Key Features

*   **Flexible Gate Definitions:** Create custom and standard quantum gates.
*   **Parameterized Circuits:** Use symbolic variables to build dynamic circuits.
*   **Circuit Optimization & Compilation:** Transform, optimize, and compile circuits for efficient execution.
*   **Hardware Device Modeling:**  Model and simulate quantum hardware characteristics.
*   **Noise Modeling:** Incorporate noise effects to realistically simulate quantum systems.
*   **Built-in Simulators:** Utilize multiple built-in simulators for circuit execution.
*   **High-Performance Simulation with qsim:** Integrate with qsim for fast simulations.
*   **Interoperability:** Seamlessly integrate with NumPy and SciPy.
*   **Cross-Platform Compatibility:** Runs on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq requires Python 3.11 or higher. Comprehensive installation instructions are available in the [Cirq documentation](https://quantumai.google/cirq/start/install).

## Quick Start - "Hello Qubit" Example

Get started quickly by running a simple quantum simulation.

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

This example demonstrates the basics of creating a circuit, running it, and interpreting the results.

## Documentation and Resources

Explore the comprehensive documentation and resources to master Cirq.

### Tutorials

*   [Video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4) - Learn Cirq through engaging video content.
*   [Jupyter notebook-based tutorials](https://colab.research.google.com/github/quantumlib/Cirq) - Experiment with Cirq directly in your browser.
*   [Text-based tutorials](https://quantumai.google/cirq) - Dive deep with detailed, step-by-step tutorials.

### Reference Documentation

*   [Current stable release](https://quantumai.google/reference/python/cirq/all_symbols) -  Understand the current stable version.
*   [Pre-release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly) - Explore the newest features.

### Examples

*   [Examples subdirectory](./examples/) - Discover practical applications in the Cirq GitHub repository.
*   [Experiments page](https://quantumai.google/cirq/experiments/) - Find advanced examples.

### Change Log

*   [Cirq releases](https://github.com/quantumlib/cirq/releases) - Stay up to date with release notes and changes.

## Integrations

Enhance your Cirq experience with these complementary tools from Google Quantum AI:

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

Join the vibrant Cirq community and contribute to the future of quantum computing.

<a href="https://github.com/quantumlib/Cirq/graphs/contributors"><img
width="150em" alt="Total number of contributors to Cirq"
src="https://img.shields.io/github/contributors/quantumlib/cirq?label=Contributors&logo=github&color=ccc&style=flat-square"/></a>

Cirq thrives on the contributions of over 200 individuals.  We encourage open and inclusive participation, adhering to our [code of conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md).

### Stay Informed

Stay updated with Cirq developments through:

*   `cirq-announce`:  Sign up for the [low-volume mailing list](https://groups.google.com/forum/#!forum/cirq-announce) for releases and major announcements.
*   GitHub notifications: Configure [repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications) for Cirq.
*   Atom/RSS feed: Subscribe to the GitHub [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom) to monitor releases.
*   PyPI RSS feed:  Track Cirq releases via the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).

### Engage and Discuss

*   Ask and answer questions about Cirq on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) and tag them with [`cirq`](https://quantumcomputing.stackexchange.com/questions/tagged/cirq).
*   Join the Cirq Cynq meetings by signing up to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev) for meeting invitations.

### Contribute

*   Share your ideas and report issues by [opening an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose).
*   Contribute to the project by reviewing the [list of good first issues](https://github.com/quantumlib/Cirq/contribute), reading the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md), and submitting [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

Cite Cirq in your publications to give credit where it's due. Use the Zenodo DOI for the specific version used.

<div align="center">
  [![Download BibTeX bibliography record for latest Cirq
  release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
  [![Download CSL JSON bibliography record for latest Cirq
  release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)
</div>

For all releases, find the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any questions or concerns, contact the maintainers at quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>