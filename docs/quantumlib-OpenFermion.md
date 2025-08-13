<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

[![Python](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/OpenFermion/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c)](https://pypi.org/project/OpenFermion)
[![PyPI Downloads](https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads)](https://img.shields.io/pypi/dm/OpenFermion)

## OpenFermion: Simulate and Analyze Fermionic Systems for Quantum Computing

OpenFermion is an open-source library designed for quantum algorithm compilation and analysis, offering tools to represent and manipulate fermionic and qubit Hamiltonians for simulating quantum chemistry and other fermionic systems.

**[Explore the OpenFermion Repository](https://github.com/quantumlib/OpenFermion)**

### Key Features

*   **Fermionic Hamiltonian Manipulation:** Efficiently work with fermionic operators and their representations.
*   **Quantum Algorithm Compilation:** Compile and analyze quantum algorithms for simulating fermionic systems.
*   **Integration with Quantum Simulators:** Supports various quantum simulators and electronic structure packages.
*   **Extensive Tooling:** Includes data structures and tools for manipulating fermionic and qubit Hamiltonians.
*   **Open Source:**  Freely available for research and commercial use.

### Get Started

**Installation:**

Install the latest stable release using pip:

```bash
pip install openfermion
```

**Documentation:**

*   [Official Documentation](https://quantumai.google/openfermion)
*   [Installation Guide](https://quantumai.google/openfermion/install)
*   [API Reference](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Developer Installation:**

For the latest development version:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
pip install -e .
```

### Plugins for Extended Functionality

OpenFermion's functionality is extended through various plugins:

**High-Performance Simulators:**

*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): High-performance fermionic quantum evolution emulator.

**Circuit Compilation Plugins:**

*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion):  Integration with Rigetti's Forest.
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Integration with Xanadu's Strawberry Fields.

**Electronic Structure Package Plugins:**

*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4):  Integration with Psi4.
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Integration with PySCF.
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Integration with DIRAC.
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Integration with Q-Chem.

### Contributing

We welcome contributions!  Please review the [contribution guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md) and sign the Contributor License Agreement (CLA) before submitting pull requests.  Use GitHub issues to report bugs or suggest improvements.

### Authors

(See original README for a comprehensive list of authors.)

### How to Cite

Please cite the following paper when using OpenFermion in your research:

*   Jarrod R McClean, et al.  *OpenFermion: The Electronic Structure Package for Quantum Computers*.  Quantum Science and Technology 5.3 (2020): 034014.  [https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta](https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta)

### Disclaimer

This is not an official Google product.