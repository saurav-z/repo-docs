<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

<div align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License: Apache 2.0"></a>
  <a href="https://pypi.org/project/OpenFermion"><img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Release"></a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads"></a>
</div>

## OpenFermion: Simulate and Analyze Quantum Algorithms for Fermionic Systems

OpenFermion is an open-source library designed for quantum simulation, providing tools to compile, analyze, and manipulate fermionic and qubit Hamiltonians, particularly for quantum chemistry applications.  [Explore the OpenFermion Repository](https://github.com/quantumlib/OpenFermion).

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:** Efficiently work with data structures representing fermionic Hamiltonians.
*   **Quantum Algorithm Compilation:** Tools for preparing and compiling quantum circuits.
*   **Integration with Quantum Simulators:** Seamless integration with high-performance simulators and electronic structure packages.
*   **Open Source & Community Driven:** Benefit from a vibrant community and contribute to cutting-edge quantum research.

**Installation and Documentation**

Install the latest stable release using pip:

```bash
python -m pip install --user openfermion
```

For detailed documentation and installation guides, visit:

*   **Documentation:** [quantumai.google/openfermion](https://quantumai.google/openfermion)
*   **Installation:** [quantumai.google/openfermion/install](https://quantumai.google/openfermion/install)
*   **API Docs:** [quantumai.google/reference/python/openfermion/all_symbols](https://quantumai.google/reference/python/openfermion/all_symbols)
*   **Tutorials:** [quantumai.google/openfermion/tutorials/intro_to_openfermion](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Supported Platforms:**

OpenFermion is tested on Mac, Windows, and Linux.  Note that electronic structure plugins are primarily compatible with Mac and Linux. A Docker image is available for alternative installations.

**Developer Installation:**

To install the latest development version:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

**Plugins**

OpenFermion uses plugins for extended functionality.

**High-performance simulators:**

*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): Emulator of fermionic quantum evolutions.

**Circuit compilation plugins:**

*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): Integration with Rigetti's Forest.
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Integration with Xanadu's Strawberry Fields.

**Electronic structure package plugins:**

*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Integration with Psi4.
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Integration with PySCF.
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Integration with DIRAC.
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Integration with Q-Chem.

**How to Contribute**

Contributions are welcome! See the [CONTRIBUTING guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md) for details.

**Authors**

[List of Authors](https://github.com/quantumlib/OpenFermion#authors)

**How to cite**

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

**Disclaimer**

Copyright 2017 The OpenFermion Developers.
This is not an official Google product.