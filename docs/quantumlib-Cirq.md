# Cirq: Build, Manipulate, and Run Quantum Circuits (Powered by Google Quantum AI)

**Cirq is a powerful Python library for designing, simulating, and executing quantum circuits.**  Learn more and contribute on the [original Cirq repo](https://github.com/quantumlib/Cirq).

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

## Key Features

*   **Flexible Gate Definitions:** Define custom quantum gates to match your specific needs.
*   **Parameterized Circuits:** Use symbolic variables to create dynamic and adaptable circuits.
*   **Circuit Transformation & Optimization:**  Transform, compile, and optimize circuits for performance.
*   **Hardware Device Modeling:** Model the behavior of real-world quantum hardware.
*   **Noise Modeling:** Simulate the effects of noise in quantum systems.
*   **Multiple Built-in Simulators:** Utilize various simulators for circuit execution and testing.
*   **Integration with qsim:** Leverage high-performance quantum circuit simulation.
*   **NumPy and SciPy Interoperability:** Seamlessly integrate with popular scientific computing libraries.
*   **Cross-Platform Compatibility:** Works on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq supports Python 3.11 and later.  Refer to the [Cirq documentation](https://quantumai.google/cirq/start/install) for detailed installation instructions.

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

## Documentation

*   **[Cirq Home Page](https://quantumai.google/cirq):** The central hub for Cirq documentation and resources.
*   **Tutorials:**
    *   [Video Tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4)
    *   [Jupyter Notebook Tutorials](https://colab.research.google.com/github/quantumlib/Cirq)
    *   [Text-based Tutorials](https://quantumai.google/cirq)
*   **Reference Documentation:** [Current Stable Release](https://quantumai.google/reference/python/cirq/all_symbols) & [Pre-Release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)
*   **Examples:**  Explore the [examples subdirectory](./examples/) and the [Experiments page](https://quantumai.google/cirq/experiments/) on the Cirq documentation site.
*   **Change Log:** [Cirq Releases](https://github.com/quantumlib/cirq/releases) on GitHub.

## Integrations

Explore Google Quantum AI's suite of open-source software for enhanced Cirq functionality:

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

Cirq thrives on community contributions and collaboration.

*   **Contributors:**  [Contributions](https://github.com/quantumlib/Cirq/graphs/contributors) by 200+ people.
*   **Code of Conduct:**  [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Stay Informed

*   **Announcements:** Sign up to the low-volume mailing list [`cirq-announce`] for major updates.
*   **Releases:**
    *   GitHub notifications: Configure [repository notifications].
    *   Atom/RSS: Subscribe to the [Cirq releases Atom feed].
    *   RSS from PyPI: Subscribe to the [PyPI releases RSS feed].

### Engage & Discuss

*   **Questions:** Post questions to the [Quantum Computing Stack Exchange] and tag them with [`cirq`].
*   **Discussions:** Join the biweekly virtual meeting, _Cirq Cynq_ (sign up to [_cirq-dev_]).

### Contribute

*   **Feature Requests & Bug Reports:** [Open an issue on GitHub].
*   **Develop Cirq Code:** Review the [list of good first issues], read the [contribution guidelines], and open [pull requests].

[Open an issue on GitHub]: https://github.com/quantumlib/Cirq/issues/new/choose
[list of good first issues]: https://github.com/quantumlib/Cirq/contribute
[contribution guidelines]: https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md
[pull requests]: https://help.github.com/articles/about-pull-requests
[`cirq-announce`]: https://groups.google.com/forum/#!forum/cirq-announce
[repository notifications]: https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications
[Cirq releases Atom feed]: https://github.com/quantumlib/Cirq/releases.atom
[PyPI releases RSS feed]: https://pypi.org/rss/project/cirq/releases.xml
[Quantum Computing Stack Exchange]: https://quantumcomputing.stackexchange.com
[`cirq`]: https://quantumcomputing.stackexchange.com/questions/tagged/cirq
[_cirq-dev_]: https://groups.google.com/forum/#!forum/cirq-dev

## Citing Cirq

Cite the Cirq version you use in your publications.  Use the following links to download the bibliographic record for the latest stable release:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

For citations and records in other formats, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For inquiries, email quantum-oss-maintainers@google.com.

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
Key improvements and changes:

*   **SEO Optimization:**  Incorporated relevant keywords like "quantum circuits," "quantum simulation," and "quantum computing" in headings and text.
*   **Clear Structure:** Uses headings, subheadings, and bullet points for readability and easy navigation.
*   **Concise Summary:** Starts with a strong one-sentence hook to capture attention.
*   **Actionable Links:**  Provides direct links to the GitHub repository, documentation, tutorials, and community resources.
*   **Comprehensive Feature List:** Highlights the core capabilities of Cirq.
*   **Updated Information:** Includes the latest Python version support.
*   **Community Focus:**  Emphasizes community involvement and how to contribute.
*   **Citation Information:**  Makes it easy to cite Cirq in research papers.
*   **Simplified Formatting:** Uses Markdown for cleaner presentation.
*   **Removed redundant information:** Streamlined sections to reduce clutter
*   **Clear Section Titles:**  Uses clear and concise section titles.
*   **Improved Readability:**  Enhanced readability with better formatting and sentence structure.
*   **Removed repetitive links.**
*   **Included all links:** Made sure all links in the original README were preserved.