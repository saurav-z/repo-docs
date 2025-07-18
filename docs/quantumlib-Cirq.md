<div align="center">
<img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

## Cirq: Build, Manipulate, and Run Quantum Circuits with Python

**Cirq is a Python library designed for creating, manipulating, and running quantum circuits, empowering researchers and developers to explore the world of quantum computing.**  [Explore the Cirq repository](https://github.com/quantumlib/Cirq).

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

**Key Features:**

*   **Flexible Gate Definitions:** Define custom quantum gates to model specific hardware.
*   **Parameterized Circuits:** Build circuits with symbolic variables for greater flexibility.
*   **Circuit Transformation and Optimization:**  Transform and optimize circuits for efficient execution.
*   **Hardware Device Modeling:** Simulate and model quantum hardware characteristics.
*   **Noise Modeling:** Simulate the effects of noise in quantum circuits.
*   **Built-in Simulators:** Utilize multiple built-in quantum circuit simulators.
*   **High-Performance Simulation with qsim:** Integrate with qsim for high-performance circuit simulation.
*   **Interoperability:** Seamlessly integrates with NumPy and SciPy.
*   **Cross-Platform Compatibility:** Works on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq supports Python 3.11 and later.  For comprehensive installation instructions, see the [Cirq installation guide](https://quantumai.google/cirq/start/install).

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

## Cirq Documentation

Access a wealth of resources to learn and master Cirq:

*   **Tutorials:**
    *   [Video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4)
    *   [Jupyter notebook-based tutorials](https://colab.research.google.com/github/quantumlib/Cirq)
    *   [Text-based tutorials](https://quantumai.google/cirq)
*   **Reference Documentation:**
    *   [Current stable release](https://quantumai.google/reference/python/cirq/all_symbols)
    *   [Pre-release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)
*   **Examples:**
    *   [Examples subdirectory](./examples/) in the Cirq GitHub repo.
    *   [Experiments page](https://quantumai.google/cirq/experiments/)

## Integrations

Enhance your quantum computing workflow with these Google Quantum AI software integrations:

| Your interests                                  | Software to explore  |
|-------------------------------------------------|----------------------|
| Quantum algorithms?<br>Fault-tolerant quantum computing (FTQC)? | [Qualtran](https://github.com/quantumlib/qualtran) |
| Large circuits and/or a lot of simulations?     | [qsim](https://github.com/quantumlib/qsim) |
| Circuits with thousands of qubits and millions of Clifford operations? | [Stim](https://github.com/quantumlib/stim) |
| Quantum error correction (QEC)?                 | [Stim](https://github.com/quantumlib/stim) |
| Chemistry and/or material science?              | [OpenFermion](https://github.com/quantumlib/openfermion)<br>[OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)<br>[OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF)<br>[OpenFermion-Psi4](https://github.com/quantumlib/OpenFermion-Psi4) |
| Quantum machine learning (QML)?                 | [TensorFlow Quantum](https://github.com/tensorflow/quantum) |
| Real experiments using Cirq?                    | [ReCirq](https://github.com/quantumlib/ReCirq) |

## Community

Cirq thrives on community contributions.

*   **Contributors:**  Over 200 contributors. [View contributors](https://github.com/quantumlib/Cirq/graphs/contributors)
*   **Code of Conduct:** [Read the code of conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Announcements

Stay updated on Cirq developments:

*   **Mailing List:** [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce) for releases and major announcements.
*   **GitHub Notifications:** Configure [repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications) for Cirq.
*   **Atom/RSS:** Subscribe to the GitHub [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom).
*   **PyPI:** Subscribe to the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml)

### Questions and Discussions

*   **Quantum Computing Stack Exchange:** Ask questions on [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) and tag them with [`cirq`](https://quantumcomputing.stackexchange.com/questions/tagged/cirq).
*   **Cirq Cynq:** Join _Cirq Cynq_, the biweekly virtual meeting of contributors. Sign up to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev) for meeting invitations.

### Contributions

*   **Feature requests/bug reports:** [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose).
*   **Develop Cirq code:** Explore the [list of good first issues](https://github.com/quantumlib/Cirq/contribute), review the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md), and open [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

Cite Cirq in your publications to properly credit the project.  Download bibliographic records for the latest release:

<div align="center">
[![Download BibTeX bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)
</div>

For all releases and citation formats, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For assistance, please email quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product.  It is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>