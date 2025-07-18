<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
  <br>
  <br>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License: Apache 2.0">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Version">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads">
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads">
</div>

## OpenFermion: The Open-Source Library for Quantum Simulation of Fermionic Systems

OpenFermion is a powerful, open-source library designed for simulating fermionic systems, including applications in quantum chemistry, by providing tools for representing, manipulating, and analyzing fermionic and qubit Hamiltonians.  ([See the original repository](https://github.com/quantumlib/OpenFermion))

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:** Provides data structures and tools for creating, modifying, and analyzing fermionic Hamiltonians.
*   **Qubitization:** Supports the conversion of fermionic Hamiltonians to qubit representations, essential for quantum computing.
*   **Circuit Compilation:** Offers tools for compiling and optimizing quantum circuits for simulating fermionic systems.
*   **Integration with Quantum Chemistry Packages:** Includes plugins for seamless integration with popular electronic structure packages like Psi4 and PySCF.
*   **Extensible Architecture:** Relies on modular plugin libraries for expanded functionality, allowing users to tailor the library to their needs.
*   **High-Performance Simulators:** Offers high-performance emulators like OpenFermion-FQE for simulating fermionic quantum evolutions.

## Installation and Documentation

Install the latest stable version of OpenFermion using pip:

```bash
python -m pip install --user openfermion
```

For development installations and comprehensive documentation, explore the following resources:

*   **Documentation:** [quantumai.google/openfermion](https://quantumai.google/openfermion)
*   **Installation:** [quantumai.google/openfermion/install](https://quantumai.google/openfermion/install)
*   **API Docs:** [quantumai.google/reference/python/openfermion/all_symbols](https://quantumai.google/reference/python/openfermion/all_symbols)
*   **Tutorials:** [quantumai.google/openfermion/tutorials/intro_to_openfermion](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

### Developer Installation
To install the latest development version:
```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

## Plugins

OpenFermion leverages plugins to extend functionality. Find out more below!

### High-Performance Simulators

*   **OpenFermion-FQE:** A high-performance emulator for fermionic quantum evolutions. [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)

### Circuit Compilation Plugins

*   **Forest-OpenFermion:** Supports integration with Forest.
*   **SFOpenBoson:** Supports integration with Strawberry Fields.

### Electronic Structure Package Plugins

*   **OpenFermion-Psi4:** For use with Psi4.
*   **OpenFermion-PySCF:** For use with PySCF.
*   **OpenFermion-Dirac:** For use with DIRAC.
*   **OpenFermion-QChem:** For use with Q-Chem.

## How to Contribute

Contributions are welcome! Please refer to the guidelines in the original README for submitting contributions, including CLA requirements, code style, and testing.

*   **GitHub Issues:** [https://github.com/quantumlib/OpenFermion/issues](https://github.com/quantumlib/OpenFermion/issues)
*   **Stack Exchange:** Questions are welcome on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/) with the tag "openfermion".

## Authors

The authors of OpenFermion are listed in the original README.

## How to Cite

To cite OpenFermion, please reference the following paper:

```
Jarrod R McClean, et al.
*OpenFermion: The Electronic Structure Package for Quantum Computers*.
Quantum Science and Technology 5.3 (2020): 034014
```

## Disclaimer

This is not an official Google product.