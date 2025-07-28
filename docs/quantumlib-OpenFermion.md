<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

[![Python](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/OpenFermion/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c)](https://pypi.org/project/OpenFermion)
[![Downloads](https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads)](https://img.shields.io/pypi/dm/OpenFermion)

# OpenFermion: Simulate Fermionic Systems for Quantum Algorithms

OpenFermion is a powerful open-source library designed to help researchers and developers compile and analyze quantum algorithms for simulating fermionic systems, including applications in quantum chemistry and materials science. Explore the possibilities of quantum simulation with **[OpenFermion](https://github.com/quantumlib/OpenFermion)**.

## Key Features:

*   **Fermionic Hamiltonian Manipulation:** Provides data structures and tools for representing and manipulating fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Facilitates the compilation of quantum algorithms for simulating fermionic systems.
*   **Integration with Quantum Computing Platforms:** Supports integration with various quantum computing platforms and electronic structure packages through modular plugins.
*   **Open-Source and Accessible:** Free to use, modify, and distribute under the Apache 2.0 license.

## Installation and Documentation

### Installation

#### Prerequisites:
Ensure you have Python 3.10 or higher and pip installed.

#### Install via pip:
```bash
python -m pip install --user openfermion
```

#### Developer Installation:
```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

### Documentation

Comprehensive documentation, tutorials, and API references are available to help you get started:

*   **Documentation:** [quantumai.google/openfermion](https://quantumai.google/openfermion)
*   **Installation Guide:** [quantumai.google/openfermion/install](https://quantumai.google/openfermion/install)
*   **API Reference:** [quantumai.google/reference/python/openfermion/all_symbols](https://quantumai.google/reference/python/openfermion/all_symbols)
*   **Tutorials:** [quantumai.google/openfermion/tutorials/intro_to_openfermion](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

## Plugins

OpenFermion's modular design allows for integration with various tools and platforms through plugins:

### High-Performance Simulators
*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): A high-performance emulator of fermionic quantum evolutions.

### Circuit Compilation Plugins

*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): For integration with Forest.
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): For integration with Strawberry Fields.

### Electronic Structure Package Plugins

*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): For integration with Psi4.
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): For integration with PySCF.
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): For integration with DIRAC.
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): For integration with Q-Chem.

## How to Contribute

We welcome contributions! Please review the contribution guidelines before submitting pull requests.

## Authors

(List of Authors - as provided in the original README)

## How to Cite

If you use OpenFermion in your research, please cite the following:

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

## Disclaimer

Copyright 2017 The OpenFermion Developers. This is not an official Google product.