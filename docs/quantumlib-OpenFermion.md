<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

<div align="center">
  <br>
  <a href="https://www.python.org/downloads/" target="_blank"><img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License: Apache 2.0"></a>
  <a href="https://pypi.org/project/OpenFermion" target="_blank"><img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/OpenFermion" target="_blank"><img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads"></a>
</div>

<br>

**OpenFermion is an open-source library designed to simplify the compilation and analysis of quantum algorithms for simulating fermionic systems, including quantum chemistry.**

## Key Features

*   **Fermionic Hamiltonian Manipulation:** Build and manipulate fermionic and qubit Hamiltonians with ease.
*   **Quantum Algorithm Compilation:** Compile and analyze quantum algorithms for simulating fermionic systems.
*   **Electronic Structure Integration:** Integrates with popular electronic structure packages for quantum chemistry calculations.
*   **Modular Plugin Architecture:** Extensible design for supporting a wide range of quantum simulators and electronic structure packages.
*   **Cross-Platform Compatibility:** Supports Mac, Windows, and Linux (with electronic structure plugins primarily compatible on Mac/Linux).
*   **Docker Support:** Provides a Docker image for easy setup and usage on any operating system, including Windows.

## Installation and Documentation

Get started with OpenFermion today!

*   **Documentation:**  [quantumai.google/openfermion](https://quantumai.google/openfermion)
*   **Installation:** [quantumai.google/openfermion/install](https://quantumai.google/openfermion/install)
*   **API Documentation:** [quantumai.google/reference/python/openfermion/all_symbols](https://quantumai.google/reference/python/openfermion/all_symbols)
*   **Tutorials:** [quantumai.google/openfermion/tutorials/intro_to_openfermion](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

### Installation Methods

**Prerequisites:**  Ensure you have Python 3.10+ and an up-to-date version of pip.

**1. Library Install (PyPI):**

```bash
python -m pip install --user openfermion
```

**2. Developer Install (from source):**

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

### Plugins

OpenFermion utilizes plugins for expanded functionality:

*   **High-Performance Simulators:**
    *   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): High-performance emulator for fermionic quantum evolutions.

*   **Circuit Compilation Plugins:**
    *   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): Integration with Forest.
    *   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Integration with Strawberry Fields.

*   **Electronic Structure Package Plugins:**
    *   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Integration with Psi4.
    *   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Integration with PySCF.
    *   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Integration with DIRAC.
    *   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Integration with Q-Chem.

## How to Contribute

We welcome contributions! Please review our [contribution guidelines](https://github.com/quantumlib/OpenFermion) and adhere to the following:

*   **Contributor License Agreement (CLA):** Required for contributions.  Sign the CLA at [https://cla.developers.google.com/](https://cla.developers.google.com/).
*   **Pull Requests:**  Submit contributions via GitHub pull requests.
*   **Testing:** Ensure your code includes comprehensive tests.
*   **Style Guide:** Follow PEP 8 guidelines.
*   **Documentation:** Provide thorough documentation for all new code.
*   **Issues and Questions:**  Use [GitHub issues](https://github.com/quantumlib/OpenFermion/issues) for bug reports and feature requests.  For general questions, use the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/) with the 'openfermion' tag.

## Authors

A list of authors is available in the original [README](https://github.com/quantumlib/OpenFermion).

## How to Cite

If you use OpenFermion in your research, please cite:

Jarrod R McClean, Nicholas C Rubin, Kevin J Sung, Ian D Kivlichan, Xavier Bonet-Monroig,
Yudong Cao, Chengyu Dai, E Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale,
Thomas Häner, Tarini Hardikar, Vojtĕch Havlíček, Oscar Higgott, Cupjin Huang, Josh Izaac,
Zhang Jiang, Xinle Liu, Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman,
Isil Ozfidan, Maxwell D Radin, Jhonathan Romero, Nicolas P D Sawaya, Bruno Senjean,
Kanav Setia, Sukin Sim, Damian S Steiger, Mark Steudtner, Qiming Sun, Wei Sun, Daochen Wang,
Fang Zhang, and Ryan Babbush
*OpenFermion: The Electronic Structure Package for Quantum Computers*.
`Quantum Science and Technology 5.3 (2020): 034014 <https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta>`__.

---

**[Back to OpenFermion Repository](https://github.com/quantumlib/OpenFermion)**
```
Key improvements and SEO optimizations:

*   **Clear, Concise Hook:** The one-sentence summary immediately grabs the reader's attention.
*   **Keyword Optimization:** Includes relevant keywords like "quantum algorithms," "fermionic systems," "quantum chemistry," "Hamiltonians," and plugin names throughout.
*   **Headings:**  Uses clear, descriptive headings for each section, improving readability and organization.
*   **Bulleted Key Features:**  Highlights the main selling points of the library in an easy-to-scan format.
*   **Links:**  Provides direct links to important resources (documentation, installation, API, tutorials, original repo), making it easy for users to find information.
*   **Installation Instructions:**  Offers both pip and developer installation options with clear instructions.
*   **Plugin Emphasis:**  Clearly explains the plugin architecture and lists available plugins with links.
*   **Contribution & Citation Guidance:**  Provides concise instructions for contributing and citing the library.
*   **Concise Language:** Uses clear and straightforward language for better comprehension.
*   **Author List:** Includes author information.
*   **Back to Repo Link:** Easy navigation back to the original repo.
*   **Image Alt Tags:** Good practice, although the original README already had them.