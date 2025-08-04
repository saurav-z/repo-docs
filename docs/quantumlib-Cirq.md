<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: A Python Library for Quantum Computing

**Cirq** is a powerful Python package that allows you to design, manipulate, and run quantum circuits on quantum computers and simulators. Build quantum programs with ease using Cirq!

[![Licensed under the Apache 2.0
license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and
higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on
PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in
Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

**[View the Cirq Repository on GitHub](https://github.com/quantumlib/Cirq)**

## Key Features

*   **Flexible Gate Definitions:** Define custom quantum gates tailored to your needs.
*   **Parameterized Circuits:** Utilize symbolic variables for dynamic circuit construction.
*   **Circuit Optimization & Compilation:** Transform, compile, and optimize circuits for efficient execution.
*   **Hardware Device Modeling:** Model the behavior of quantum hardware devices.
*   **Noise Modeling:** Simulate the effects of noise in quantum systems.
*   **Built-in Simulators:** Access multiple quantum circuit simulators for testing and experimentation.
*   **High-Performance Simulation with qsim:** Seamlessly integrate with [qsim](https://github.com/quantumlib/qsim) for faster simulations.
*   **NumPy and SciPy Interoperability:** Easily integrate with popular scientific Python libraries.
*   **Cross-Platform Compatibility:** Works on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq requires Python 3.11 or later. Refer to the [Installation Guide](https://quantumai.google/cirq/start/install) for detailed instructions.

## Quick Start – “Hello Qubit” Example

Get started with Cirq by running a simple quantum simulation:

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

This will output a circuit and the results of the simulation, demonstrating a basic Cirq workflow.

## Documentation & Tutorials

Explore the comprehensive [Cirq documentation](https://quantumai.google/cirq) for detailed information and tutorials:

*   **Video Tutorials:** Learn Cirq through engaging [video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4).
*   **Jupyter Notebook Tutorials:** Experiment with Cirq directly in your browser using [Jupyter notebooks](https://colab.research.google.com/github/quantumlib/Cirq).
*   **Text-Based Tutorials:** Dive into the [text-based tutorials](https://quantumai.google/cirq) for in-depth guidance.

### Additional Documentation Resources

*   **Reference Documentation:** Access the [current stable release documentation](https://quantumai.google/reference/python/cirq/all_symbols) and the [pre-release documentation](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly).
*   **Examples:** Find practical code examples in the [examples subdirectory](./examples/) of the Cirq GitHub repo and on the [Experiments page](https://quantumai.google/cirq/experiments/).
*   **Change Log:** Review the [Cirq releases](https://github.com/quantumlib/Cirq/releases) page for the latest updates.

## Integrations

Enhance your Cirq projects with these Google Quantum AI open-source software integrations:

| Your Interests                                  | Software to Explore                      |
|-------------------------------------------------|------------------------------------------|
| Quantum algorithms & FTQC?                       | [Qualtran](https://github.com/quantumlib/qualtran)                               |
| Large circuits or simulations?               | [qsim](https://github.com/quantumlib/qsim)                                      |
| Thousands of qubits, Clifford operations?    | [Stim](https://github.com/quantumlib/stim)                                         |
| Quantum error correction?                      | [Stim](https://github.com/stim)                                                  |
| Chemistry / Material Science?                  | [OpenFermion](https://github.com/quantumlib/openfermion), [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE), [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF), [OpenFermion-Psi4](https://github.com/quantumlib/OpenFermion-Psi4) |
| Quantum machine learning?                      | [TensorFlow Quantum](https://github.com/tensorflow/quantum)                          |
| Real experiments with Cirq?                    | [ReCirq](https://github.com/quantumlib/ReCirq)                                      |

## Community

Cirq thrives on its vibrant and inclusive community.

*   **Contributors:** Over 200 contributors have helped build Cirq.  See the [Contributors](https://github.com/quantumlib/Cirq/graphs/contributors) page.
*   **Code of Conduct:** We are committed to an open and inclusive community and have a [code of conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md).

### Stay Updated

*   **Announcements:** Subscribe to the [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce) mailing list for major updates.
*   **Releases:** Follow releases via [GitHub notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications), the [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom), or the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).

### Engage & Contribute

*   **Questions:** Ask questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) with the [`cirq` tag](https://quantumcomputing.stackexchange.com/questions/tagged/cirq).
*   **Discussions:** Join the _Cirq Cynq_ biweekly meeting by signing up to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev).
*   **Feature Requests & Bug Reports:** [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose).
*   **Code Contributions:** Review the [list of good first issues](https://github.com/quantumlib/Cirq/contribute), read the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md), and submit [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

When citing Cirq in publications, reference the specific version used. Download the bibliographic record for the latest stable release:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

Find citations for all Cirq releases on the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For assistance or inquiries, contact us at quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. The project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>