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
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI version">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" >
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="Downloads">
  </a>
</div>

## OpenFermion: Your Gateway to Quantum Chemistry Simulations

**OpenFermion** is an open-source library for simulating fermionic systems on quantum computers, empowering researchers with tools for quantum chemistry and beyond. ([See the original repository](https://github.com/quantumlib/OpenFermion))

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:**  Data structures and tools for representing and manipulating fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:**  Tools to help compile and analyze quantum algorithms designed to simulate fermionic systems.
*   **Integration with Leading Packages:** Plugins for popular electronic structure packages like Psi4, PySCF, and Q-Chem.
*   **High-Performance Simulation:**  Provides OpenFermion-FQE, a high-performance emulator of fermionic quantum evolutions that exploits fermionic symmetries.
*   **Circuit Compilation Support:** Integrations with Forest and Strawberry Fields for circuit compilation.

**Get Started:**

*   **Documentation:** [quantumai.google/openfermion](https://quantumai.google/openfermion)
    *   [Installation](https://quantumai.google/openfermion/install)
    *   [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
    *   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Installation:**

*   **Stable Release (via pip):**

    ```bash
    python -m pip install --user openfermion
    ```
*   **Developer Install:**

    ```bash
    git clone https://github.com/quantumlib/OpenFermion
    cd OpenFermion
    python -m pip install -e .
    ```

**Plugins:**

OpenFermion's modular design allows for flexible integration with various tools. Key plugins include:

*   **Simulators:**
    *   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): High-performance fermionic quantum evolution emulator.

*   **Circuit Compilation:**
    *   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): Integration with Forest.
    *   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Integration with Strawberry Fields.

*   **Electronic Structure Packages:**
    *   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Integration with Psi4.
    *   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Integration with PySCF.
    *   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Integration with DIRAC.
    *   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Integration with Q-Chem.

**Contribute:**

We welcome contributions!  Please review our [contribution guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md) and sign the Contributor License Agreement (CLA) before submitting pull requests.

**Citing OpenFermion:**

If you use OpenFermion in your research, please cite the following paper:

```
Jarrod R McClean, Nicholas C Rubin, Kevin J Sung, Ian D Kivlichan, Xavier Bonet-Monroig,
Yudong Cao, Chengyu Dai, E Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale,
Thomas Häner, Tarini Hardikar, Vojtěch Havlíček, Oscar Higgott, Cupjin Huang, Josh Izaac,
Zhang Jiang, Xinle Liu, Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman,
Isil Ozfidan, Maxwell D Radin, Jhonathan Romero, Nicolas P D Sawaya, Bruno Senjean,
Kanav Setia, Sukin Sim, Damian S Steiger, Mark Steudtner, Qiming Sun, Wei Sun, Daochen Wang,
Fang Zhang, and Ryan Babbush
*OpenFermion: The Electronic Structure Package for Quantum Computers*.
`Quantum Science and Technology 5.3 (2020): 034014 <https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta>`__.
```

**Authors:**

See the original README for a full list of authors.

**Disclaimer:**

Copyright 2017 The OpenFermion Developers. This is not an official Google product.