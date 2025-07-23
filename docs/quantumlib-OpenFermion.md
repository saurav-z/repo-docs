<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

[![Python](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/OpenFermion/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c)](https://pypi.org/project/OpenFermion)
[![Downloads](https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads)](https://img.shields.io/pypi/dm/OpenFermion)

## OpenFermion: Accelerate Quantum Chemistry and Fermionic System Simulations

OpenFermion is an open-source library designed to streamline the process of compiling and analyzing quantum algorithms, offering powerful tools for simulating fermionic systems, including quantum chemistry applications.  [Learn more at the original repository](https://github.com/quantumlib/OpenFermion).

### Key Features

*   **Fermionic and Qubit Hamiltonian Manipulation:**  Provides data structures and tools for obtaining, representing, and manipulating fermionic and qubit Hamiltonians.
*   **Modular Plugin Architecture:** Integrates with a variety of plugins for circuit compilation, electronic structure calculations, and high-performance simulations.
*   **High-Performance Simulators:** Offers tools like OpenFermion-FQE to efficiently emulate fermionic quantum evolutions.
*   **Integration with Quantum Computing Platforms:**  Includes plugins for Forest, Strawberry Fields, and more.
*   **Electronic Structure Package Integration:** Supports plugins for Psi4, PySCF, DIRAC, and Q-Chem.

### Getting Started

**Installation**

Install the latest stable version of OpenFermion using pip:

```bash
python -m pip install --user openfermion
```

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

**Documentation**

Comprehensive documentation is available to guide you:

*   [Installation](https://quantumai.google/openfermion/install)
*   [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Run Examples:**

Interact with OpenFermion through interactive Jupyter Notebooks in [Colab](https://colab.research.google.com/github/quantumlib/OpenFermion) or [MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples).

### Plugins

Extend OpenFermion's functionality with these plugins:

#### High-performance simulators
*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)

#### Circuit compilation plugins
*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion)
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson)

#### Electronic structure package plugins
*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4)
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF)
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac)
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem)

### Contributing

Contributions are welcome! Please review the [contribution guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md) to get started, including signing a Contributor License Agreement (CLA).

### Authors
See the full list of [Authors](https://github.com/quantumlib/OpenFermion/blob/main/AUTHORS) on the repository.

### How to Cite

When using OpenFermion for research projects, please cite:

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

### Disclaimer

Copyright 2017 The OpenFermion Developers.
This is not an official Google product.