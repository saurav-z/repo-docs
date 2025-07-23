<div align="center">

<img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">

</div>

## Cirq: Build and Simulate Quantum Circuits with Python

**Cirq is a powerful Python package for creating, manipulating, and simulating quantum circuits, enabling researchers and developers to explore the forefront of quantum computing.** ([Original Repo](https://github.com/quantumlib/Cirq))

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

*   **Flexible Gate Definitions:** Define and customize quantum gates easily.
*   **Parameterized Circuits:** Create circuits with symbolic variables for enhanced flexibility.
*   **Circuit Transformation & Optimization:**  Tools for compilation, optimization, and efficient execution.
*   **Hardware Device Modeling:** Simulate and analyze circuits considering hardware characteristics.
*   **Noise Modeling:**  Incorporate noise models to simulate real-world quantum hardware behavior.
*   **Built-in Simulators:** Includes multiple simulators for quantum circuit execution.
*   **High-Performance Simulation with qsim:** Integrates with qsim for fast and efficient simulations.
*   **Seamless Integration:** Works with NumPy and SciPy.
*   **Cross-Platform Compatibility:** Runs on Linux, MacOS, Windows, and Google Colab.

**Sections:**

*   [Features](#features)
*   [Installation](#installation)
*   [Quick Start](#quick-start--hello-qubit-example)
*   [Documentation](#cirq-documentation)
*   [Integrations](#integrations)
*   [Community](#community)
*   [Citing Cirq](#citing-cirq)
*   [Contact](#contact)

## Installation

Cirq supports Python version 3.11 and later and can be used on Linux, macOS, Windows, and Google Colab. Detailed installation instructions are available in the [Cirq documentation](https://quantumai.google/cirq/start/install).

## Quick Start – “Hello Qubit” Example

Get started with Cirq quickly by running the following example in your Python interpreter:

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

This will output a simulated result, demonstrating the basic functionality of Cirq.

## Cirq Documentation

Comprehensive documentation is available on the [Cirq home page](https://quantumai.google/cirq) and includes:

### Tutorials

*   [Video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4)
*   [Jupyter notebook-based tutorials](https://colab.research.google.com/github/quantumlib/Cirq)
*   [Text-based tutorials](https://quantumai.google/cirq) with [installation] instructions, [basics], and guides to [build] and [simulate] circuits.

### Reference Documentation

*   [Current stable release documentation](https://quantumai.google/reference/python/cirq/all_symbols)
*   [Pre-release documentation](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)

### Examples

*   Explore practical applications in the [examples subdirectory](./examples/) of the Cirq GitHub repo.
*   Find more examples on the [Experiments page](https://quantumai.google/cirq/experiments/).

### Change log

*   Review changes in each release on the [Cirq releases](https://github.com/quantumlib/cirq/releases) page.

## Integrations

Cirq integrates with a suite of open-source software from Google Quantum AI, enhancing its capabilities:

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

Cirq has a vibrant and active community with over [200 contributors](https://github.com/quantumlib/Cirq/graphs/contributors). Join us!  We have a [code of conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md).

### Announcements

Stay informed about Cirq developments:

*   **Mailing list:** [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce) for releases and major announcements.
*   **GitHub notifications:** Configure [repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications) for Cirq.
*   **Atom/RSS feeds:** Subscribe to the [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom) or the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).

### Questions and Discussions

*   Ask questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) and tag them with [`cirq`].
*   Join _Cirq Cynq_, our biweekly meeting, by subscribing to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev).

### Contributions

*   [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose) to report bugs or suggest features.
*   Contribute code by reviewing the [list of good first issues](https://github.com/quantumlib/Cirq/contribute), following our [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md), and opening [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

To cite Cirq in your publications, use the bibliographic record for the latest stable release, available in various formats:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

Find all records on the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any questions or concerns, please contact quantum-oss-maintainers@google.com.

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
Key improvements:

*   **SEO Optimization:**  Includes keywords like "quantum circuits," "quantum computing," and "Python" in headings and descriptions.
*   **One-Sentence Hook:** Added a concise and engaging opening sentence.
*   **Clear Structure:**  Uses headings, subheadings, and bullet points for readability.
*   **Concise Summaries:**  Reduced the length while retaining key information.
*   **Emphasis on Benefits:**  Highlights the advantages of using Cirq.
*   **Direct Links:**  Provides links to the original repo and key resources.
*   **Complete Coverage:**  Addresses all sections of the original README, reorganized for clarity.
*   **Concise Language:** Removes unnecessary phrasing to improve readability.
*   **Stronger Call to Action:** Encourages contribution and community engagement.
*   **Clearer Formatting:** Uses Markdown effectively for visual appeal and accessibility.
*   **Adds `Installation` section**, which was missing from the original.