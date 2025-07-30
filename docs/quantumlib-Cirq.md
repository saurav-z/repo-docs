<div align="center">
  <img width="300px" alt="Cirq logo" src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# Cirq: Build and Simulate Quantum Circuits with Python

Cirq is a powerful Python library for building, manipulating, and simulating quantum circuits, empowering researchers and developers to explore the potential of quantum computing. ([View the original repository](https://github.com/quantumlib/Cirq))

[![Licensed under the Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.11 and higher](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://img.shields.io/badge/dynamic/json?label=OpenSSF&logo=springsecurity&logoColor=white&style=flat-square&colorA=gray&colorB=d56420&suffix=%25&query=$.badge_percentage_0&uri=https://bestpractices.coreinfrastructure.org/projects/10063.json)](https://www.bestpractices.dev/projects/10063)
[![Cirq project on PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

**Key Features:**

*   **Flexible Gate Definitions:** Define custom quantum gates to match your specific needs.
*   **Parameterized Circuits:** Build circuits with symbolic variables for enhanced flexibility.
*   **Circuit Optimization & Compilation:** Transform, optimize and compile quantum circuits for efficient execution.
*   **Hardware Device Modeling:** Accurately model the characteristics of real quantum hardware.
*   **Noise Modeling:** Simulate the effects of noise in quantum systems.
*   **Multiple Simulators:** Utilize built-in simulators for various quantum circuit simulations.
*   **Integration with qsim:** Leverage qsim for high-performance simulation.
*   **NumPy & SciPy Interoperability:** Seamlessly integrate with these essential Python libraries.
*   **Cross-Platform Compatibility:** Compatible with Linux, macOS, Windows, and Google Colab.

## Getting Started

### Installation

Cirq supports Python 3.11 and later.  Installation is straightforward; detailed instructions can be found in the [Installation section](https://quantumai.google/cirq/start/install) of the online documentation.

### Quickstart: "Hello Qubit" Example

Here's a basic example to get you started with Cirq:

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

*   **Main Documentation:** [Cirq Home Page](https://quantumai.google/cirq) on the Quantum AI website
*   **Tutorials:**
    *   [Video Tutorials](https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4)
    *   [Jupyter Notebook Tutorials](https://colab.research.google.com/github/quantumlib/Cirq)
    *   [Text-Based Tutorials](https://quantumai.google/cirq)
*   **Reference Documentation:**
    *   [Current Stable Release](https://quantumai.google/reference/python/cirq/all_symbols)
    *   [Pre-release](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly)
*   **Examples:**
    *   [Cirq Examples Directory](./examples/)
    *   [Experiments Page](https://quantumai.google/cirq/experiments/)
*   **Change Log:** [Cirq Releases](https://github.com/quantumlib/cirq/releases)

## Integrations

Enhance your Cirq experience with these Google Quantum AI open-source tools:

<div align="center">

| Your interests                                  | Software to explore  |
|-------------------------------------------------|----------------------|
| Quantum algorithms?<br>Fault-tolerant quantum computing (FTQC)? | [Qualtran](https://github.com/quantumlib/qualtran) |
| Large circuits and/or a lot of simulations?     | [qsim](https://github.com/quantumlib/qsim) |
| Circuits with thousands of qubits and millions of Clifford operations? | [Stim](https://github.com/quantumlib/stim) |
| Quantum error correction (QEC)?                 | [Stim](https://github.com/quantumlib/stim) |
| Chemistry and/or material science?              | [OpenFermion](https://github.com/quantumlib/openfermion)<br>[OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)<br>[OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF)<br>[OpenFermion-Psi4](https://github.com/quantumlib/OpenFermion-Psi4) |
| Quantum machine learning (QML)?                 | [TensorFlow Quantum](https://github.com/tensorflow/quantum) |
| Real experiments using Cirq?                    | [ReCirq](https://github.com/quantumlib/ReCirq) |

</div>

## Community

Join the Cirq community and contribute to the future of quantum computing!

*   **Contributors:** [See all contributors](https://github.com/quantumlib/Cirq/graphs/contributors)
*   **Code of Conduct:** [View the Code of Conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)
*   **Announcements:**
    *   [`cirq-announce` Mailing List](https://groups.google.com/forum/#!forum/cirq-announce)
    *   [GitHub Releases Atom Feed](https://github.com/quantumlib/Cirq/releases.atom)
    *   [PyPI Releases RSS Feed](https://pypi.org/rss/project/cirq/releases.xml)
*   **Discussions:**
    *   [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com) (tag: `cirq`)
    *   Join _Cirq Cynq_, our biweekly virtual meeting of contributors. Sign up to [_cirq-dev_] to get an automatic meeting invitation!
*   **Contributions:**
    *   [Open an issue on GitHub](https://github.com/quantumlib/Cirq/issues/new/choose)
    *   [List of good first issues](https://github.com/quantumlib/Cirq/contribute)
    *   [Contribution Guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md)
    *   [Pull Requests](https://help.github.com/articles/about-pull-requests)

## Citing Cirq

Please cite the Cirq version you use in your publications.

<div align="center">

[![Download BibTeX bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

For more citation options, visit the [Cirq page on Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For inquiries, contact: quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product.  This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2019 The Cirq Developers.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>
```
Key improvements and SEO considerations:

*   **Clear Title and Introduction:**  The title and one-sentence hook immediately tell the user what Cirq is and its purpose.
*   **Keyword Optimization:**  Uses relevant keywords like "quantum circuits," "quantum computing," "simulation," and "Python library" throughout the description.
*   **Organized Structure:** Uses clear headings and subheadings for readability and SEO.
*   **Bulleted Key Features:**  Highlights the key features in a concise, easily scannable format.
*   **Contextual Links:**  Links to relevant resources and documentation sections, improving user experience and SEO.
*   **Emphasis on Community:**  Highlights community engagement and contributions.
*   **Citation Information:**  Clearly explains how to cite Cirq, which helps with discoverability.
*   **Contact Information:** Provides contact information.
*   **Disclaimer:** Includes a standard disclaimer.
*   **Clean Formatting:** Improved markdown for better readability.