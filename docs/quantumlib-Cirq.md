<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Build and Simulate Quantum Circuits with Python

**Cirq is a powerful Python library enabling researchers and developers to design, manipulate, and simulate quantum circuits, paving the way for advancements in quantum computing.** [View the original repository](https://github.com/quantumlib/Cirq).

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

## Key Features

*   **Flexible Gate Definitions:** Define custom quantum gates to match your specific needs.
*   **Parameterized Circuits:**  Utilize symbolic variables for creating and manipulating complex quantum circuits.
*   **Circuit Transformation and Optimization:**  Transform, compile, and optimize circuits for improved performance.
*   **Hardware Device Modeling:** Model real-world quantum hardware characteristics, including noise.
*   **Noise Modeling:**  Simulate the effects of noise on quantum circuits.
*   **Built-in Simulators:** Includes multiple simulators for various quantum circuit simulations.
*   **High-Performance Simulation with qsim:**  Integrates with qsim for high-speed simulations.
*   **NumPy and SciPy Interoperability:** Seamlessly integrates with NumPy and SciPy for data analysis and manipulation.
*   **Cross-Platform Compatibility:** Works on Linux, MacOS, Windows, and Google Colab.

## Installation

Cirq is compatible with Python 3.11 and later.  Refer to the [installation instructions](https://quantumai.google/cirq/start/install) for detailed instructions on how to get started.

## Quick Start – "Hello Qubit" Example

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

## Documentation & Resources

*   **[Cirq Home Page](https://quantumai.google/cirq):** The main resource for documentation and information.
*   **Tutorials:**
    *   [Video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4)
    *   [Jupyter Notebook-based tutorials](https://colab.research.google.com/github/quantumlib/Cirq)
    *   [Text-based tutorials](https://quantumai.google/cirq)
*   **Reference Documentation:**
    *   [Current Stable Release](https://quantumai.google/reference/python/cirq/all_symbols)
    *   [Pre-release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)
*   **Examples:**
    *   [Examples Subdirectory](./examples/) in the Cirq GitHub repo
    *   [Experiments Page](https://quantumai.google/cirq/experiments/) on the Cirq documentation site
*   **Change Log:** [Cirq Releases](https://github.com/quantumlib/cirq/releases)

## Integrations

Cirq integrates with several open-source software packages from Google Quantum AI to enhance your quantum computing workflows:

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

Cirq thrives on a vibrant community of contributors.

[![Total number of contributors to Cirq](https://img.shields.io/github/contributors/quantumlib/cirq?label=Contributors&logo=github&color=ccc&style=flat-square)](https://github.com/quantumlib/Cirq/graphs/contributors)

*   **Code of Conduct:** [CODE_OF_CONDUCT.md](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)
*   **Announcements:**
    *   [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce) mailing list for releases and major announcements.
    *   [GitHub Notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications) for releases.
    *   [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom)
    *   [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml)
*   **Discussions:**
    *   [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) with the `cirq` tag.
    *   Join _Cirq Cynq_, the biweekly meeting of contributors (sign up to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev)).
*   **Contributions:**
    *   [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose) for feature requests or bug reports.
    *   [List of good first issues](https://github.com/quantumlib/Cirq/contribute) for contributing code.
    *   [Contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md)
    *   [Pull requests](https://help.github.com/articles/about-pull-requests)

## Citing Cirq

To cite Cirq in your work:

<div align="center">
[![Download BibTeX bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)
</div>

For formatted citations and records for all releases, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For questions or concerns, email quantum-oss-maintainers@google.com.

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
Key changes and improvements:

*   **SEO-Optimized Title and Description:** Included "Quantum Circuits" and other relevant keywords in the title and description.
*   **Clear Structure:**  Used headings and subheadings to organize the information.
*   **Bulleted Key Features:**  Presented the features in a clear, easy-to-read bulleted list.
*   **Concise Language:** Streamlined the language for better readability.
*   **Call to action:** Included a one-sentence summary "Cirq is a powerful Python library enabling researchers and developers to design, manipulate, and simulate quantum circuits, paving the way for advancements in quantum computing"
*   **Internal Linking:** Kept links to relevant sections (Installation, Quick Start, Documentation, etc.)
*   **Complete Rewrite:**  Revised the entire README for better clarity and searchability.
*   **Community Section Enhancement:** Expanded the Community section to provide more avenues for engagement.
*   **Citing Cirq:** Added a section on how to cite Cirq.