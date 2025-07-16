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

## OpenFermion: Accelerate Quantum Chemistry and Fermionic System Simulations

**OpenFermion** is an open-source library designed to facilitate the development of quantum algorithms for simulating fermionic systems, offering tools for representing and manipulating fermionic and qubit Hamiltonians. Explore the power of quantum simulation for complex calculations!

[View the original repository](https://github.com/quantumlib/OpenFermion)

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:** Data structures and tools to obtain, represent, and manipulate fermionic and qubit Hamiltonians.
*   **Integration with Quantum Simulators:** Easily compile and analyze quantum algorithms for simulating fermionic systems.
*   **Quantum Chemistry Focus:** Specifically designed to assist in simulating quantum chemistry problems.
*   **Modular Plugin Architecture:** Extensible with plugins for circuit compilation and electronic structure calculations.
*   **Cross-Platform Compatibility:** Tested on Mac, Windows, and Linux. Docker image available for consistent environment.

**Key Integrations:**

*   **High-Performance Simulators:** OpenFermion-FQE
*   **Circuit Compilation:** Forest-OpenFermion, SFOpenBoson
*   **Electronic Structure Packages:** OpenFermion-Psi4, OpenFermion-PySCF, OpenFermion-Dirac, OpenFermion-QChem

**Getting Started**

*   **Documentation:** [quantumai.google/openfermion](https://quantumai.google/openfermion)
    *   [Installation](https://quantumai.google/openfermion/install)
    *   [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
    *   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Installation**

Install the latest stable release using pip:

```bash
python -m pip install --user openfermion
```

**Developer Installation:**

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

**Contribute**

We welcome contributions! Please review the [contribution guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md) and ensure you have signed a Contributor License Agreement (CLA).

**How to Cite**

If you use OpenFermion in your research, please cite:

```
Jarrod R McClean, Nicholas C Rubin, Kevin J Sung, Ian D Kivlichan, Xavier Bonet-Monroig,
Yudong Cao, Chengyu Dai, E Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale,
Thomas Häner, Tarini Hardikar, Vojtĕch Havlíček, Oscar Higgott, Cupjin Huang, Josh Izaac,
Zhang Jiang, Xinle Liu, Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman,
Isil Ozfidan, Maxwell D Radin, Jhonathan Romero, Nicolas P D Sawaya, Bruno Senjean,
Kanav Setia, Sukin Sim, Damian S Steiger, Mark Steudtner, Qiming Sun, Wei Sun, Daochen Wang,
Fang Zhang, and Ryan Babbush
*OpenFermion: The Electronic Structure Package for Quantum Computers*.
Quantum Science and Technology 5.3 (2020): 034014.
```

**Authors**

See the original README for a complete list of authors.

**Disclaimer**

Copyright 2017 The OpenFermion Developers.
This is not an official Google product.