<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Build and Simulate Quantum Circuits with Python

**Cirq is a Python library designed for writing, manipulating, and running quantum circuits, empowering researchers and developers to explore the cutting edge of quantum computing.** ([View on GitHub](https://github.com/quantumlib/Cirq))

[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![DOI](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

## Key Features

*   **Flexible Gate Definitions:** Define and customize quantum gates.
*   **Parameterized Circuits:** Utilize symbolic variables for circuit design.
*   **Circuit Transformation and Optimization:** Compile and optimize circuits for efficient execution.
*   **Hardware Device Modeling:** Model and simulate quantum hardware.
*   **Noise Modeling:** Simulate realistic quantum system behavior.
*   **Built-in Simulators:** Access multiple quantum circuit simulators.
*   **High-Performance Simulation:** Integrate with [qsim](https://github.com/quantumlib/qsim) for accelerated simulations.
*   **Seamless Integration:** Interoperability with [NumPy](https://numpy.org) and [SciPy](https://scipy.org).
*   **Cross-Platform Compatibility:** Run on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq supports Python 3.11 and later.  Refer to the [installation guide](https://quantumai.google/cirq/start/install) for detailed instructions.

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

Explore the comprehensive documentation on the [Cirq home page](https://quantumai.google/cirq).

### Tutorials

*   **Video Tutorials:** Learn Cirq visually via YouTube.
*   **Jupyter Notebook Tutorials:** Interactive tutorials in your browser via Google Colab.
*   **Text-Based Tutorials:** Step-by-step guides for building and simulating circuits, ideal for local installations.

### Reference Documentation

*   [Current Stable Release](https://quantumai.google/reference/python/cirq/all_symbols)
*   [Pre-Release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)

### Examples

*   [Examples Subdirectory](./examples/) in the GitHub repository with a variety of applications.
*   [Experiments Page](https://quantumai.google/cirq/experiments/) with advanced examples.

### Change Log

*   [Cirq Releases](https://github.com/quantumlib/cirq/releases)

## Integrations

Leverage the Google Quantum AI software stack for advanced quantum computing workflows.

| Your Interests                                  | Software to Explore                                       |
|-------------------------------------------------|------------------------------------------------------------|
| Quantum Algorithms, Fault-Tolerant Quantum Computing (FTQC) | [Qualtran](https://github.com/quantumlib/qualtran)           |
| Large Circuits / Many Simulations                | [qsim](https://github.com/quantumlib/qsim)                 |
| Circuits with thousands of qubits / millions of Clifford operations | [Stim](https://github.com/quantumlib/stim)                   |
| Quantum Error Correction (QEC)                    | [Stim](https://github.com/quantumlib/stim)                   |
| Chemistry / Material Science                      | [OpenFermion](https://github.com/quantumlib/openfermion), [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE), [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF), [OpenFermion-Psi4](https://github.com/quantumlib/OpenFermion-Psi4) |
| Quantum Machine Learning (QML)                    | [TensorFlow Quantum](https://github.com/tensorflow/quantum) |
| Real Experiments                                | [ReCirq](https://github.com/quantumlib/ReCirq)              |

## Community

<a href="https://github.com/quantumlib/Cirq/graphs/contributors"><img
width="150em" alt="Total number of contributors to Cirq"
src="https://img.shields.io/github/contributors/quantumlib/cirq?label=Contributors&logo=github&color=ccc&style=flat-square"/></a>

Cirq thrives with the contributions of over 200 individuals. We encourage participation in our inclusive community.

*   [Contributions](https://github.com/quantumlib/Cirq/graphs/contributors)
*   [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Stay Informed

*   **Announcements:** Join the low-volume [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce) mailing list.
*   **Releases (GitHub):** Configure [repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications) or subscribe to the [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom).
*   **Releases (PyPI):** Subscribe to the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).

### Engage

*   **Questions:** Ask questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) and tag them with [`cirq`].
*   **Discussions:** Join _Cirq Cynq_, our biweekly virtual meeting. Subscribe to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev) for invitations.

### Contribute

*   **Feature Requests/Bug Reports:** [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose).
*   **Code Contributions:** Review the [list of good first issues](https://github.com/quantumlib/Cirq/contribute), read the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md), and submit [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

When citing Cirq in your work, please cite the version you use.  Download the bibliographic record for the latest stable release:

<div align="center">

[![Download BibTeX](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

Find citations for all releases on the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

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