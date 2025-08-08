<div align="center">

<img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">

</div>

# Cirq: Your Gateway to Quantum Computing in Python

**Cirq is a powerful Python library for creating, manipulating, and simulating quantum circuits, making quantum computing accessible and enabling cutting-edge research.** ([View on GitHub](https://github.com/quantumlib/Cirq))

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

*   **Flexible Gate Definitions:** Define and customize quantum gates to match your research needs.
*   **Parameterized Circuits:** Build circuits with symbolic variables for dynamic and adaptable quantum programs.
*   **Circuit Transformation, Compilation, and Optimization:** Streamline your circuits for better performance.
*   **Hardware Device Modeling:** Simulate and optimize for the realities of quantum hardware.
*   **Noise Modeling:** Accurately simulate the effects of noise in your quantum circuits.
*   **Built-in Simulators:** Explore a range of simulators for your quantum circuits.
*   **High-Performance Simulation:** Integration with [qsim](https://github.com/quantumlib/qsim) for rapid simulation.
*   **Seamless Integration:** Interoperability with [NumPy](https://numpy.org) and [SciPy](https://scipy.org).
*   **Cross-Platform Compatibility:** Works on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq is easy to install and use. It supports Python 3.11 and later and runs on various platforms, including Google Colab. Find complete installation instructions in the [Cirq documentation](https://quantumai.google/cirq/start/install).

## Quick Start: "Hello Qubit" Example

Get started quickly with this simple example.  After installing Cirq, open a Python interpreter and run the following code:

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

This will output the circuit and simulation results. Explore the [many Cirq tutorials](https://quantumai.google/cirq) to continue learning.

## Documentation

The primary source for Cirq documentation is the [Cirq home page on the Quantum AI website](https://quantumai.google/cirq).

### Tutorials

*   [Video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4)
*   [Jupyter notebook-based tutorials](https://colab.research.google.com/github/quantumlib/Cirq)
*   [Text-based tutorials](https://quantumai.google/cirq)

### Reference Documentation

*   [Current stable release](https://quantumai.google/reference/python/cirq/all_symbols)
*   [Pre-release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)

### Examples

*   [Examples subdirectory](./examples/)
*   [Experiments page](https://quantumai.google/cirq/experiments/)

### Change Log

*   [Cirq releases](https://github.com/quantumlib/cirq/releases)

## Integrations

Google Quantum AI offers a suite of open-source tools to enhance your Cirq experience:

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

Cirq thrives on community contributions.

<a href="https://github.com/quantumlib/Cirq/graphs/contributors"><img
width="150em" alt="Total number of contributors to Cirq"
src="https://img.shields.io/github/contributors/quantumlib/cirq?label=Contributors&logo=github&color=ccc&style=flat-square"/></a>

*   [Contributions](https://github.com/quantumlib/Cirq/graphs/contributors) by over 200 individuals.
*   [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md) promotes an inclusive environment.

### Stay Informed

*   [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce) mailing list for major announcements.
*   [Repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications) on GitHub.
*   [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom).
*   [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml)

### Engage

*   Ask questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) using the `cirq` tag.
*   Join the biweekly _Cirq Cynq_ meeting by subscribing to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev).

### Contribute

*   [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose) for feature requests or bug reports.
*   Review the [list of good first issues](https://github.com/quantumlib/Cirq/contribute), read the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md), and submit [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

Cite Cirq in your publications using the Zenodo DOI for the specific version you used:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

Find formatted citations and records for all releases on the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any inquiries, please email quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>