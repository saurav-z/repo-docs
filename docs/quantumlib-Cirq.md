<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Quantum Computing for the NISQ Era

Cirq is a powerful Python library for building, manipulating, and running quantum circuits, empowering researchers and developers to explore the potential of quantum computing.  ([Original Repo](https://github.com/quantumlib/Cirq))

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

**Key Features:**

*   **Flexible Gate Definitions:** Define and customize quantum gates.
*   **Parameterized Circuits:**  Create circuits using symbolic variables for greater flexibility.
*   **Circuit Optimization:** Transform, compile, and optimize circuits for efficient execution.
*   **Hardware Device Modeling:**  Model quantum hardware to improve results on real devices.
*   **Noise Modeling:** Simulate and analyze the effects of noise in quantum systems.
*   **Built-in Simulators:**  Utilize multiple simulators for quantum circuit execution.
*   **Integration with QSim:** Leverage the high-performance [qsim](https://github.com/quantumlib/qsim) simulator.
*   **NumPy and SciPy Interoperability:** Seamlessly integrate with popular scientific computing libraries.
*   **Cross-Platform Compatibility:**  Run Cirq on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq supports Python 3.11 and later.  Refer to the [installation instructions](https://quantumai.google/cirq/start/install) for detailed setup steps.

## Quick Start - "Hello Qubit" Example

Get started with Cirq quickly with this example:

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

This will output the circuit and the results of the simulation.

## Documentation & Resources

Comprehensive documentation, tutorials, and examples are available to help you learn and use Cirq:

### Tutorials
*   [Video tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4)
*   [Jupyter notebook-based tutorials](https://colab.research.google.com/github/quantumlib/Cirq)
*   [Text-based tutorials](https://quantumai.google/cirq)

### Reference Documentation

*   [Current stable release](https://quantumai.google/reference/python/cirq/all_symbols)
*   [Pre-release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)

### Examples
*   [Examples Subdirectory](./examples/)
*   [Experiments page](https://quantumai.google/cirq/experiments/)

### Change Log

*   [Cirq releases](https://github.com/quantumlib/Cirq/releases)

## Integrations

Cirq integrates seamlessly with a suite of open-source tools to enhance your quantum computing workflow:

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

Cirq thrives on community contributions.  Join the vibrant community and contribute to the future of quantum computing!

*   [Contributors](https://github.com/quantumlib/Cirq/graphs/contributors)
*   [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Stay Updated

*   **Announcements:** Subscribe to the low-volume mailing list [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce).
*   **Releases:**
    *   GitHub notifications: configure [repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications) for Cirq.
    *   Atom/RSS:  Subscribe to the [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom).
    *   PyPI: Subscribe to the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).

### Get Involved

*   **Questions & Discussions:** Post your questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) and tag them with [`cirq`](https://quantumcomputing.stackexchange.com/questions/tagged/cirq).  Join _Cirq Cynq_, the biweekly virtual meeting of contributors. Sign up to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev) to get an invitation.
*   **Contributions:** [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose), check the [list of good first issues](https://github.com/quantumlib/Cirq/contribute), read the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md), and then submit [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

When citing Cirq in your publications, please use the Zenodo DOI for the latest release.  Download the bibliographic record in your preferred format:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

For all releases, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any questions or concerns, please email quantum-oss-maintainers@google.com.

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
Key improvements and SEO considerations:

*   **Clear Title and Introduction:** Starts with a concise, SEO-friendly title ("Cirq: Quantum Computing for the NISQ Era") and a one-sentence hook that immediately grabs attention and highlights the library's purpose.
*   **Keyword Optimization:**  Includes relevant keywords like "quantum computing," "quantum circuits," "NISQ era," "Python library," and core features, improving search engine visibility.
*   **Structured Content:** Uses headings (H2, H3) to organize information logically, making it easier for users to scan and understand.
*   **Bulleted Lists:**  Employs bulleted lists for the "Key Features" and to present other key information in a clear and concise manner, which is also SEO-friendly.
*   **Internal and External Linking:**  Provides links to relevant documentation sections, tutorials, examples, and related projects.  Links back to the original repo.
*   **Emphasis on Benefits:** Highlights the advantages of using Cirq (e.g., flexible gate definitions, circuit optimization, hardware modeling) to attract users.
*   **Call to Action:** Encourages users to get involved in the community (e.g., "Get Involved" section).
*   **Clean Formatting:**  Uses markdown for clear and readable formatting.
*   **SEO Metadata (Implicit):** The headings and structure implicitly guide search engines to understand the content and its relevance to specific queries.
*   **Conciseness:** Removes unnecessary wording and keeps the explanations focused.
*   **Community Focus:** Highlights the active community and ways to contribute.