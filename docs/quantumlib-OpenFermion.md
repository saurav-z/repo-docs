<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion logo" width="75%">
</div>

<div align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Compatible with Python versions 3.10 and higher">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="Licensed under the Apache 2.0 license">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="OpenFermion project on PyPI">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads">
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="OpenFermion downloads per month from PyPI">
  </a>
</div>

**OpenFermion: Your Open-Source Toolkit for Quantum Simulation of Fermionic Systems**

OpenFermion is an open-source library designed to simplify the process of simulating fermionic systems, especially in quantum chemistry, by providing data structures and tools for manipulating fermionic and qubit Hamiltonians.

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:** Tools for representing, manipulating, and analyzing fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:**  Compilation and analysis tools for quantum algorithms.
*   **Integration with Quantum Computing Platforms:** Supports integration with various quantum computing platforms and electronic structure packages.
*   **Modular Plugin Architecture:** Extensible through plugins for expanded functionality.

**Get Started:**

*   **[Original Repository](https://github.com/quantumlib/OpenFermion)**
*   **Interactive Examples:** Run interactive Jupyter Notebooks in [Colab](https://colab.research.google.com/github/quantumlib/OpenFermion) or [MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples).

**Installation and Documentation**

To get started with OpenFermion, you'll need to install it using `pip`. Ensure you have an up-to-date version of pip.

*   **Documentation:**  [quantumai.google/openfermion](https://quantumai.google/openfermion)
*   **Installation Guide:** [Installation](https://quantumai.google/openfermion/install)
*   **API Reference:** [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
*   **Tutorials:** [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Installation Options**

OpenFermion is tested on Mac, Windows, and Linux.

*   **Stable Release:**
    ```bash
    python -m pip install --user openfermion
    ```

*   **Developer Install:**
    ```bash
    git clone https://github.com/quantumlib/OpenFermion
    cd OpenFermion
    python -m pip install -e .
    ```

**Docker Installation (Recommended for Windows):** A Docker image is available with OpenFermion and select plugins pre-installed.  See the [docker folder](https://github.com/quantumlib/OpenFermion/tree/master/docker) for instructions.

**Plugins**

OpenFermion's modular design relies on plugins for key functionalities.

*   **High-Performance Simulators:**
    *   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)
*   **Circuit Compilation Plugins:**
    *   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion)
    *   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson)
*   **Electronic Structure Package Plugins:**
    *   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4)
    *   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF)
    *   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac)
    *   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem)

**Contributing**

Contributions are welcome!  Please review the [contribution guidelines](https://github.com/quantumlib/OpenFermion/blob/master/CONTRIBUTING.md) and ensure you have signed a Contributor License Agreement (CLA).

**Get Involved:**

*   **Issues:**  [Github issues](https://github.com/quantumlib/OpenFermion/issues)
*   **Questions:**  Post questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/) with the 'openfermion' tag.

**Authors:**
*(List of Authors - same as original)*

**How to cite**
*(Citation Info - same as original)*

**Disclaimer:**
*(Disclaimer - same as original)*