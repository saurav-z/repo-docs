<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Build, Manipulate, and Simulate Quantum Circuits with Python

Cirq is a powerful Python library enabling you to design, manipulate, and run quantum circuits for research and development.

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

[**View the Cirq Repository on GitHub**](https://github.com/quantumlib/Cirq)

## Key Features

*   **Flexible Gate Definitions:** Define custom quantum gates to match your specific needs.
*   **Parameterized Circuits:** Utilize symbolic variables for creating adaptable circuit designs.
*   **Circuit Transformation and Optimization:** Streamline and optimize your circuits for better performance.
*   **Hardware Device Modeling:** Accurately simulate quantum hardware characteristics.
*   **Noise Modeling:** Incorporate noise models to simulate real-world quantum system behavior.
*   **Multiple Built-in Simulators:** Test and analyze circuits with different simulation approaches.
*   **High-Performance Simulation with qsim:** Integrate with qsim for faster simulation speeds.
*   **NumPy and SciPy Interoperability:** Leverage the power of NumPy and SciPy for advanced data analysis and processing.
*   **Cross-Platform Compatibility:** Cirq runs seamlessly on Linux, macOS, Windows, and Google Colab.

## Installation

Cirq requires Python 3.11 or later. For detailed installation instructions, please consult the [Cirq Installation Guide](https://quantumai.google/cirq/start/install).

## Quick Start - "Hello Qubit" Example

Get started quickly with Cirq by running this simple example:

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

This will output a simple quantum circuit and its simulated results.

## Documentation and Resources

Explore the comprehensive Cirq documentation and related resources:

*   **Tutorials:**
    *   [Video Tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4)
    *   [Jupyter Notebook Tutorials](https://colab.research.google.com/github/quantumlib/Cirq)
    *   [Text-based Tutorials](https://quantumai.google/cirq)
*   **Reference Documentation:**
    *   [Current Stable Release](https://quantumai.google/reference/python/cirq/all_symbols)
    *   [Pre-release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)
*   **Examples:**
    *   [Cirq Examples Directory](./examples/)
    *   [Experiments Page](https://quantumai.google/cirq/experiments/)
*   **Change Log:**
    *   [Cirq Releases](https://github.com/quantumlib/Cirq/releases)

## Integrations

Extend Cirq's functionality with these Google Quantum AI open-source software integrations:

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

Cirq thrives on a vibrant and active community. Join us and contribute:

*   **Contributors:** [Contributors](https://github.com/quantumlib/Cirq/graphs/contributors)
*   **Code of Conduct:** [Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)

### Stay Informed

*   **Announcements:** Subscribe to the low-volume mailing list [`cirq-announce`](https://groups.google.com/forum/#!forum/cirq-announce).
*   **Releases:** Get notified via GitHub [repository notifications](https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications), the [Cirq releases Atom feed](https://github.com/quantumlib/Cirq/releases.atom), or the [PyPI releases RSS feed](https://pypi.org/rss/project/cirq/releases.xml).

### Engage and Discuss

*   **Questions and Discussions:** Post questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) using the tag [`cirq`](https://quantumcomputing.stackexchange.com/questions/tagged/cirq).
*   **Cirq Cynq:** Join the biweekly virtual meeting of contributors by subscribing to [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev).

### Contribute

*   **Feature Requests/Bug Reports:** [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose)
*   **Code Contributions:** Review the [list of good first issues](https://github.com/quantumlib/Cirq/contribute), read the [contribution guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md), and submit [pull requests](https://help.github.com/articles/about-pull-requests).

## Citing Cirq

When referencing Cirq in your work, please cite the specific version you use. Find the citation information for the latest stable release:

<div align="center">
[![Download BibTeX bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)
</div>

For citations in other formats and previous releases, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any questions or concerns, please contact quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product, and it is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>
```
Key improvements:

*   **SEO-friendly Title and Description:** Added a clear title and introductory sentence to capture attention and improve search visibility.
*   **Concise Key Features:**  Features are presented in a bulleted list for readability and easy scanning.
*   **Organized Structure:**  The README is broken down into logical sections with clear headings for easy navigation.
*   **Clear Calls to Action:**  Encourages users to explore documentation, examples, and the community.
*   **Emphasis on Benefits:** Highlights the benefits of using Cirq, such as hardware modeling and optimization.
*   **Complete Installation Guide Reference:**  Directs users to the most important installation information.
*   **Links to Resources:** Provides direct links to the source code, documentation, and community resources.
*   **SEO Optimization:** Uses relevant keywords like "quantum circuits," "quantum simulation," and "Python library."
*   **Concise and Focused Content:** Unnecessary information has been removed to improve clarity.