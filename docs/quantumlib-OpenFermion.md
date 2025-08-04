<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

<div align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+ compatibility">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License: Apache 2.0">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI version">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads">
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI downloads">
  </a>
</div>

## OpenFermion: Simulate and Analyze Quantum Algorithms for Fermionic Systems

OpenFermion is an open-source Python library designed for the simulation and analysis of quantum algorithms, particularly for quantum chemistry and fermionic systems, offering powerful tools for Hamiltonian manipulation and quantum circuit compilation.  Explore the original repository [here](https://github.com/quantumlib/OpenFermion).

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:** Efficiently work with fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:**  Tools for compiling quantum circuits.
*   **Integration with Quantum Computing Platforms:** Supports integration with platforms like Forest, Strawberry Fields, and others.
*   **Electronic Structure Plugins:** Integrates with popular electronic structure packages such as Psi4, PySCF, and Q-Chem.
*   **Cross-Platform Compatibility:** Tested on Mac, Windows, and Linux.

**Quick Links:**

*   **Documentation:** [quantumai.google/openfermion](https://quantumai.google/openfermion)
*   **Installation:** [quantumai.google/openfermion/install](https://quantumai.google/openfermion/install)
*   **API Docs:** [quantumai.google/reference/python/openfermion/all_symbols](https://quantumai.google/reference/python/openfermion/all_symbols)
*   **Tutorials:** [quantumai.google/openfermion/tutorials/intro_to_openfermion](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

## Installation

### Installing with pip

To install the latest stable release, use pip:

```bash
python -m pip install --user openfermion
```

### Developer Installation

For the latest development version:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

## Plugins

OpenFermion's functionality is extended through modular plugins:

*   **OpenFermion-FQE:** High-performance emulator for fermionic quantum evolutions ([OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)).
*   **Forest-OpenFermion:** Integration with Forest ([Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion)).
*   **SFOpenBoson:** Integration with Strawberry Fields ([SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson)).
*   **OpenFermion-Psi4:** Integration with Psi4 ([OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4)).
*   **OpenFermion-PySCF:** Integration with PySCF ([OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF)).
*   **OpenFermion-Dirac:** Integration with DIRAC ([Openfermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac)).
*   **OpenFermion-QChem:** Integration with Q-Chem ([OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem)).

## Contributing

Contributions are welcome!  See the [GitHub Help](https://help.github.com/articles/about-pull-requests/) for pull request guidelines.  Please adhere to the project's coding style (PEP 8) and include comprehensive tests and documentation.  All contributions require a Contributor License Agreement (CLA).

*   **Issues:** [Github issues](https://github.com/quantumlib/OpenFermion/issues)
*   **Questions:**  Quantum Computing Stack Exchange with the 'openfermion' tag.

## Authors

[List of authors](https://github.com/quantumlib/OpenFermion#authors)

## How to Cite

If you use OpenFermion in your research, please cite:

Jarrod R McClean, Nicholas C Rubin, Kevin J Sung, Ian D Kivlichan, Xavier Bonet-Monroig,
Yudong Cao, Chengyu Dai, E Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale,
Thomas Häner, Tarini Hardikar, Vojtěch Havlíček, Oscar Higgott, Cupjin Huang, Josh Izaac,
Zhang Jiang, Xinle Liu, Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman,
Isil Ozfidan, Maxwell D Radin, Jhonathan Romero, Nicolas P D Sawaya, Bruno Senjean,
Kanav Setia, Sukin Sim, Damian S Steiger, Mark Steudtner, Qiming Sun, Wei Sun, Daochen Wang,
Fang Zhang, and Ryan Babbush
*OpenFermion: The Electronic Structure Package for Quantum Computers*.
`Quantum Science and Technology 5.3 (2020): 034014 <https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta>`__.

## Disclaimer

Copyright 2017 The OpenFermion Developers.
This is not an official Google product.