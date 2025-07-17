<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Build, Manipulate, and Run Quantum Circuits with Python

[Cirq](https://github.com/quantumlib/Cirq) is a powerful Python package designed for building, manipulating, and executing quantum circuits on quantum computers and simulators, offering a comprehensive toolkit for quantum computing research and development.

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

**Key Features:**

*   **Flexible Gate Definitions:** Define and customize quantum gates with ease.
*   **Parameterized Circuits:** Utilize symbolic variables to build circuits.
*   **Circuit Transformation & Optimization:** Transform, compile, and optimize circuits for performance.
*   **Hardware Device Modeling:** Model quantum hardware to improve accuracy.
*   **Noise Modeling:** Simulate the effects of noise in quantum systems.
*   **Multiple Built-in Simulators:** Use a range of simulators for efficient circuit execution.
*   **qsim Integration:** Leverage high-performance simulation with [qsim](https://github.com/quantumlib/qsim).
*   **NumPy & SciPy Interoperability:** Seamlessly integrate with popular scientific computing libraries.
*   **Cross-Platform Compatibility:** Works on Linux, MacOS, Windows, and Google Colab.

## Installation

Cirq supports Python 3.11 and later. See the [installation instructions](https://quantumai.google/cirq/start/install) for detailed steps.

## Quick Start – "Hello Qubit" Example

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

## Documentation

Comprehensive documentation is available at the [Cirq home page](https://quantumai.google/cirq). Explore the following resources:

*   **Tutorials:**
    *   [Video Tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4)
    *   [Jupyter Notebook Tutorials](https://colab.research.google.com/github/quantumlib/Cirq)
    *   [Text-based Tutorials](https://quantumai.google/cirq)
*   **Reference Documentation:**
    *   [Current Stable Release](https://quantumai.google/reference/python/cirq/all_symbols)
    *   [Pre-release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)
*   **Examples:**
    *   [Examples Subdirectory](./examples/)
    *   [Experiments Page](https://quantumai.google/cirq/experiments/)
*   **Change Log:**
    *   [Cirq Releases](https://github.com/quantumlib/Cirq/releases)

## Integrations

Enhance your Cirq experience with these Google Quantum AI open-source tools:

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

Cirq fosters an active and inclusive community.

<a href="https://github.com/quantumlib/Cirq/graphs/contributors"><img
width="150em" alt="Total number of contributors to Cirq"
src="https://img.shields.io/github/contributors/quantumlib/cirq?label=Contributors&logo=github&color=ccc&style=flat-square"/></a>

*   **Contributors:** Over 200 contributors and growing.
*   **Code of Conduct:** Adheres to a [code of conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md).

### Stay Updated

*   **Announcements:** Subscribe to the [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce) mailing list.
*   **Releases:**
    *   GitHub notifications: configure [repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications) for Cirq.
    *   Atom/RSS: subscribe to the GitHub [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom).
    *   RSS from PyPI: subscribe to the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).

### Get Involved

*   **Questions & Discussions:** Post questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) and tag with `cirq`.
*   **Community Meetings:** Join _Cirq Cynq_, our biweekly virtual meeting. Sign up to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev) to receive meeting invitations.
*   **Contributions:**
    *   [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose) for feature requests or bug reports.
    *   Review [good first issues](https://github.com/quantumlib/Cirq/contribute) and our [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md) to start contributing code and open [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

Cite Cirq in your publications:

<div align="center">
  [![Download BibTeX bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
  [![Download CSL JSON bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)
</div>

For more citation formats, see the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any questions or concerns not addressed here, please email quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>