<div align="center">

<img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">

</div>

# Cirq: Build and Simulate Quantum Circuits with Python

**Cirq is a powerful Python library that empowers you to design, manipulate, and run quantum circuits for quantum computing research and development.** ([See on GitHub](https://github.com/quantumlib/Cirq))

[![Licensed under the Apache 2.0
license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and
higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on
PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in
Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

[Features](#features) &ndash;
[Installation](#installation) &ndash;
[Quick Start](#quick-start--hello-qubit-example) &ndash;
[Documentation](#cirq-documentation) &ndash;
[Integrations](#integrations) &ndash;
[Community](#community) &ndash;
[Citing Cirq](#citing-cirq) &ndash;
[Contact](#contact)

## Key Features

*   **Flexible Gate Definitions**: Create and customize quantum gates to model specific hardware.
*   **Parameterized Circuits**: Utilize symbolic variables for dynamic circuit design.
*   **Circuit Transformation and Optimization**: Simplify and improve circuit performance.
*   **Hardware Device Modeling**: Simulate the characteristics of real-world quantum devices.
*   **Noise Modeling**: Incorporate noise to simulate real-world quantum systems.
*   **Multiple Simulators**: Built-in simulators for various needs, including high-performance simulation with qsim.
*   **NumPy and SciPy Integration**: Seamlessly integrate with popular Python numerical libraries.
*   **Cross-Platform Compatibility**: Works on Linux, macOS, Windows, and Google Colab.

## Installation

Install Cirq easily with pip:

```bash
pip install cirq
```

Cirq supports Python 3.11 and later. For detailed installation instructions, see the [Cirq documentation](https://quantumai.google/cirq/start/install).

## Quick Start – “Hello Qubit” Example

Get started with a simple quantum simulation:

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

This code will print the circuit and the results of the simulation.

## Cirq Documentation

Explore the comprehensive [Cirq documentation](https://quantumai.google/cirq) for tutorials, reference materials, and examples.

### Tutorials

*   [Video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4) for visual learning.
*   [Jupyter notebook-based tutorials](https://colab.research.google.com/github/quantumlib/Cirq) for interactive learning.
*   [Text-based tutorials](https://quantumai.google/cirq) for a structured learning experience.

### Reference Documentation

*   Docs for the [current stable release](https://quantumai.google/reference/python/cirq/all_symbols)
*   Docs for the [pre-release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)

### Examples

*   [Examples directory](./examples/) in the GitHub repository with applications of Cirq.
*   [Experiments page](https://quantumai.google/cirq/experiments/) in the Cirq documentation.

### Change log

*   [Cirq releases](https://github.com/quantumlib/cirq/releases) on GitHub

## Integrations

Cirq integrates with several other Google Quantum AI open-source projects:

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

Cirq is a community-driven project with over 200 contributors.

[![Contributors](https://img.shields.io/github/contributors/quantumlib/cirq?label=Contributors&logo=github&color=ccc&style=flat-square)](https://github.com/quantumlib/Cirq/graphs/contributors)

*   [Contributions](https://github.com/quantumlib/Cirq/graphs/contributors)
*   [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Stay Updated

*   **Announcements**: Sign up to the low-volume mailing list [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce).
*   **Releases**:
    *   GitHub notifications: [repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications) for Cirq.
    *   Atom/RSS feed: [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom).
    *   PyPI RSS feed: [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).
    *   Cirq releases approximately every quarter.

### Engage with the Community

*   **Questions**: Post your questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) with the tag [`cirq`](https://quantumcomputing.stackexchange.com/questions/tagged/cirq).
*   **Discussions**: Join _Cirq Cynq_, the biweekly virtual meeting of contributors. Sign up to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev) for an automatic meeting invitation!

### Contribute

*   [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose) for feature requests or bug reports.
*   [List of good first issues](https://github.com/quantumlib/Cirq/contribute) to contribute to Cirq development.
*   Review the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md) and submit [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq<a name="how-to-cite-cirq"></a><a name="how-to-cite"></a>

Cite Cirq in your publications to help others reproduce your work:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

For more citation options, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For inquiries, please contact quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product.

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>