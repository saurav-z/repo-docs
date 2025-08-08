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

<br>

**OpenFermion empowers researchers and developers with tools to simulate and analyze fermionic systems for quantum computing applications, particularly in quantum chemistry.**

[Go to the Original Repository](https://github.com/quantumlib/OpenFermion)

## Key Features

*   **Hamiltonian Manipulation:** Provides data structures and tools for working with fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Facilitates the compilation of quantum algorithms for fermionic simulations.
*   **Integration with Quantum Computing Platforms:** Supports integration with various quantum computing platforms and electronic structure packages.
*   **Modular Design:** Relies on modular plugins for extended functionality, including high-performance simulators, circuit compilation, and electronic structure calculations.
*   **Open Source and Community-Driven:** An open-source library that encourages contributions from the community.

## Installation and Documentation

### Installation

Install the latest stable OpenFermion release using pip:

```bash
pip install --user openfermion
```

Alternatively, install the latest development version:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

### Documentation

Comprehensive documentation is available to help you get started and explore OpenFermion's capabilities.

*   [Installation](https://quantumai.google/openfermion/install)
*   [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

## Plugins

OpenFermion's functionality is extended through plugins. Here's a list of available plugins:

### High-Performance Simulators

*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): High-performance emulator of fermionic quantum evolutions.

### Circuit Compilation Plugins

*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): Supports integration with Forest.
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Supports integration with Strawberry Fields.

### Electronic Structure Package Plugins

*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Supports integration with Psi4.
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Supports integration with PySCF.
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Supports integration with DIRAC.
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Supports integration with Q-Chem.

## How to Contribute

We welcome contributions! Please review our contribution guidelines, including the Contributor License Agreement (CLA). Follow the standard GitHub pull request workflow, ensuring your code includes comprehensive tests and adheres to PEP 8 style guidelines.

*   **Issue Tracking:** Use [Github issues](https://github.com/quantumlib/OpenFermion/issues) to report bugs or feature requests.
*   **Questions:** Post questions to the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/) with the tag "openfermion".

## Authors

A comprehensive list of authors and contributors is included in the original README.

## How to Cite

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
Quantum Science and Technology 5.3 (2020): 034014.
```

## Disclaimer

Copyright 2017 The OpenFermion Developers.  This is not an official Google product.