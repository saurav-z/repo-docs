<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

<div align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License: Apache 2.0">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Release">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads">
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads">
  </a>
</div>

## OpenFermion: A Quantum Computing Library for Simulating Fermionic Systems

OpenFermion is a powerful open-source library designed to streamline the simulation and analysis of fermionic systems, including those relevant to quantum chemistry. Access the original repository [here](https://github.com/quantumlib/OpenFermion).

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:** Efficiently create, manipulate, and analyze fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Compile and optimize quantum algorithms for simulating fermionic systems.
*   **Integration with Quantum Computing Platforms:** Supports integration with various quantum computing platforms for circuit compilation and simulation.
*   **Plugin Architecture:** Extend functionality through a modular plugin system for specialized tasks like electronic structure calculations.
*   **Open-Source & Collaborative:** Benefit from an active community and contribute to the advancement of quantum computing.

**Getting Started**

*   **Documentation:** Access detailed documentation and tutorials at [quantumai.google/openfermion](https://quantumai.google/openfermion).

**Installation**

**Stable Release (Recommended):**
```bash
python -m pip install --user openfermion
```

**Developer Installation:**

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

**Plugins**

OpenFermion leverages plugins for extended functionalities.

**High-Performance Simulators:**

*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): High-performance emulator of fermionic quantum evolutions.

**Circuit Compilation Plugins:**

*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): Integration with Forest.
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Integration with Strawberry Fields.

**Electronic Structure Package Plugins:**

*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Integration with Psi4.
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Integration with PySCF.
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Integration with DIRAC.
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Integration with Q-Chem.

**How to Contribute**

We welcome contributions! Follow the guidelines for contributing, including the Contributor License Agreement (CLA) and code review process.

**Authors**

A comprehensive list of authors can be found in the original README.

**How to Cite**

When using OpenFermion for research, please cite:

```
Jarrod R McClean, Nicholas C Rubin, Kevin J Sung, Ian D Kivlichan, Xavier Bonet-Monroig,
Yudong Cao, Chengyu Dai, E Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale,
Thomas Häner, Tarini Hardikar, Vojtĕch Havlíček, Oscar Higgott, Cupjin Huang, Josh Izaac,
Zhang Jiang, Xinle Liu, Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman,
Isil Ozfidan, Maxwell D Radin, Jhonathan Romero, Nicolas P D Sawaya, Bruno Senjean,
Kanav Setia, Sukin Sim, Damian S Steiger, Mark Steudtner, Qiming Sun, Wei Sun, Daochen Wang,
Fang Zhang, and Ryan Babbush
*OpenFermion: The Electronic Structure Package for Quantum Computers*.
`Quantum Science and Technology 5.3 (2020): 034014 <https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta>`__.
```

**Disclaimer**

Copyright 2017 The OpenFermion Developers. This is not an official Google product.