<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Python Library for Quantum Circuit Programming

**Cirq is a powerful Python library for writing, manipulating, and running quantum circuits, enabling researchers and developers to explore the world of quantum computing.**

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

**[See the Cirq project on GitHub](https://github.com/quantumlib/Cirq)**

**Key Features:**

*   **Flexible Gate Definitions:** Define custom and standard quantum gates.
*   **Parameterized Circuits:** Build circuits with symbolic variables for advanced control.
*   **Circuit Manipulation & Optimization:** Transform, compile, and optimize circuits for efficient execution.
*   **Hardware Device Modeling:** Accurately model quantum hardware characteristics.
*   **Noise Modeling:** Simulate the effects of noise in quantum systems.
*   **Built-in Simulators:** Utilize multiple quantum circuit simulators.
*   **High-Performance Simulation:** Integrate with [qsim](https://github.com/quantumlib/qsim) for faster simulations.
*   **Interoperability:** Seamless integration with [NumPy](https://numpy.org) and [SciPy](https://scipy.org).
*   **Cross-Platform Compatibility:** Run on Linux, macOS, Windows, and [Google Colab](https://colab.google).

## Installation

Cirq supports Python 3.11 and later. For comprehensive installation instructions, refer to the [Cirq documentation](https://quantumai.google/cirq/start/install).

## Quick Start: Hello Qubit Example

Get started quickly with a simple "Hello Qubit" example:

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

## Documentation and Resources

Explore the extensive Cirq documentation for in-depth learning and examples:

*   **[Cirq Home Page](https://quantumai.google/cirq):** Your central hub for tutorials, examples, and reference documentation.
*   **Tutorials:**
    *   [Video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4)
    *   [Jupyter notebook-based tutorials](https://colab.research.google.com/github/quantumlib/Cirq)
    *   [Text-based tutorials](https://quantumai.google/cirq)
*   **Reference Documentation:**
    *   [Current stable release](https://quantumai.google/reference/python/cirq/all_symbols)
    *   [Pre-release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)
*   **Examples:**
    *   [Examples subdirectory](./examples/) in the Cirq GitHub repository
    *   [Experiments page](https://quantumai.google/cirq/experiments/) on the Cirq documentation site
*   **Change Log:**
    *   [Cirq releases](https://github.com/quantumlib/Cirq/releases)

## Integrations

Enhance your Cirq projects with the following Google Quantum AI open-source software:

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

Cirq thrives on a vibrant and collaborative community.

[![Contributors](https://img.shields.io/github/contributors/quantumlib/cirq?label=Contributors&logo=github&color=ccc&style=flat-square)](https://github.com/quantumlib/Cirq/graphs/contributors)

*   **Contributions:** [Contributions welcome!](https://github.com/quantumlib/Cirq/graphs/contributors)
*   **Code of Conduct:** [See our code of conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Stay Connected

*   **Announcements:**
    *   Sign up for the low-volume mailing list [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce).
    *   Follow Cirq releases via GitHub notifications ([repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications)),  [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom), or the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).
*   **Questions & Discussions:**
    *   Ask questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) with the `cirq` tag.
    *   Join _Cirq Cynq_, our biweekly virtual meeting of contributors.  Sign up for the  [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev) mailing list for meeting invites.
*   **Contributing:**
    *   Report issues and suggest features by [opening an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose).
    *   Contribute code by reviewing the [list of good first issues](https://github.com/quantumlib/Cirq/contribute) and reading the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md) and [opening pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

Cite Cirq in your publications to give credit and help others reproduce your results.  Download bibliographic records:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

For all releases and other formats, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any questions or concerns, please reach out to quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>