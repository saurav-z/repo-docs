<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Build, Manipulate, and Run Quantum Circuits with Python

**Cirq is a powerful Python library for quantum computing, enabling you to write, manipulate, and execute quantum circuits on both real quantum hardware and simulators.**

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

[**Get Started with Cirq**](https://github.com/quantumlib/Cirq) | [Features](#features) | [Installation](#installation) | [Quick Start](#quick-start--hello-qubit-example) | [Documentation](#cirq-documentation) | [Integrations](#integrations) | [Community](#community) | [Citing Cirq](#citing-cirq) | [Contact](#contact)

## Key Features

*   **Flexible Gate Definitions:** Define custom quantum gates easily.
*   **Parameterized Circuits:** Use symbolic variables for flexible circuit design.
*   **Circuit Transformation and Optimization:** Efficiently manipulate and optimize your circuits.
*   **Hardware Device Modeling:** Model the nuances of quantum hardware.
*   **Noise Modeling:** Simulate the effects of noise in quantum systems.
*   **Built-in Simulators:** Includes multiple quantum circuit simulators.
*   **High-Performance Simulation with qsim:** Integrate with `qsim` for fast simulation.
*   **Integration with NumPy and SciPy:** Leverage the power of Python's scientific computing libraries.
*   **Cross-Platform Compatibility:** Works on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq supports Python 3.11 and later.  For detailed installation instructions, see the [Installation Guide](https://quantumai.google/cirq/start/install) in the official documentation.

## Quick Start – “Hello Qubit” Example

Here's a simple example to get you started:

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

You should see an output similar to:

```text
Circuit:
(0, 0): ───X^0.5───M('m')───
Results:
m=11000111111011001000
```

## Cirq Documentation

Explore the comprehensive [Cirq home page](https://quantumai.google/cirq) for detailed information, tutorials, and examples.

### Tutorials

*   [Video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4) for an engaging learning experience.
*   [Jupyter notebook-based tutorials](https://colab.research.google.com/github/quantumlib/Cirq) to learn Cirq directly in your browser.
*   [Text-based tutorials](https://quantumai.google/cirq) with step-by-step instructions and practical examples.

### Reference Documentation

*   [Current stable release documentation](https://quantumai.google/reference/python/cirq/all_symbols)
*   [Pre-release documentation](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)

### Examples

*   [Examples subdirectory](./examples/) in the Cirq GitHub repository showcasing various applications.
*   [Experiments page](https://quantumai.google/cirq/experiments/) with diverse examples on the Cirq documentation site.

### Change Log

*   [Cirq releases](https://github.com/quantumlib/Cirq/releases) page on GitHub provides release notes.

## Integrations

Enhance your Cirq projects with these integrations:

<div align="center">

| Your Interests                              | Explore This Software        |
|---------------------------------------------|-----------------------------|
| Quantum Algorithms & FTQC                   | [Qualtran](https://github.com/quantumlib/qualtran)        |
| Large Circuits & Simulations                | [qsim](https://github.com/quantumlib/qsim)          |
| Thousands of Qubits & Clifford Operations   | [Stim](https://github.com/stim)          |
| Quantum Error Correction (QEC)              | [Stim](https://github.com/stim)          |
| Chemistry & Material Science               | [OpenFermion](https://github.com/quantumlib/openfermion) & related libraries |
| Quantum Machine Learning (QML)              | [TensorFlow Quantum](https://github.com/tensorflow/quantum) |
| Real Experiments with Cirq                  | [ReCirq](https://github.com/quantumlib/ReCirq)          |

</div>

## Community

Cirq thrives on community contributions!

<a href="https://github.com/quantumlib/Cirq/graphs/contributors"><img width="150em" alt="Total number of contributors to Cirq" src="https://img.shields.io/github/contributors/quantumlib/cirq?label=Contributors&logo=github&color=ccc&style=flat-square"/></a>

Cirq has over 200 contributors.  We welcome contributions!

*   [Contributions](https://github.com/quantumlib/Cirq/graphs/contributors)
*   [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Announcements

Stay informed:

*   For major announcements, subscribe to the [`cirq-announce` mailing list](https://groups.google.com/forum/#!forum/cirq-announce).
*   Follow releases via:
    *   GitHub notifications: configure [repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications).
    *   Atom/RSS: subscribe to the [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom).
    *   PyPI RSS: subscribe to the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).

### Discussions

*   Ask questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) using the `cirq` tag.
*   Join _Cirq Cynq_, our biweekly virtual meeting: sign up to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev).

### Contributions

*   [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose) for feature requests or bug reports.
*   Contribute code:
    *   Explore the [list of good first issues](https://github.com/quantumlib/Cirq/contribute).
    *   Read the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md).
    *   Submit [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

To cite Cirq, please use the Zenodo DOI for the version you are using.

<div align="center">
    [![Download BibTeX bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
    [![Download CSL JSON bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)
</div>

Find records in other formats on the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any questions or concerns not addressed here, please email quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product and is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>
```
Key improvements and SEO optimizations:

*   **Clear, Concise Title:**  "Cirq: Build, Manipulate, and Run Quantum Circuits with Python" clearly states what Cirq is.
*   **One-Sentence Hook:** The opening sentence immediately grabs attention and explains what Cirq does.
*   **Keywords:**  Includes important keywords like "quantum circuits," "quantum computing," and "Python library."
*   **Feature Highlighting:**  Uses bullet points to make key features easily scannable.
*   **Logical Structure:** The sections are clearly organized with headings and subheadings.
*   **Internal Links:**  Uses links to other parts of the document to improve user experience and SEO.
*   **External Links:**  Links to the original repository, documentation, and other resources.
*   **Concise Language:**  Uses clear and direct language throughout.
*   **Alt Text:**  Alt text is provided for images, which is good for SEO.
*   **Complete and Comprehensive:** The edited README includes all the original information but presents it in a more organized and user-friendly way.
*   **Zenodo Citation is Prominently Displayed:** The citation information is easy to find.
*   **HTML Comments Removed:**  Redundant HTML comments were removed.
*   **Direct Links to Key Resources:** Direct links to the main documentation, tutorials, examples, and the GitHub repository.