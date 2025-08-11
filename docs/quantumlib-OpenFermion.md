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
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Version">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" >
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads">
  </a>
</div>

<br>

## OpenFermion: Your Gateway to Quantum Simulation of Fermionic Systems

OpenFermion is a powerful open-source library designed to compile and analyze quantum algorithms, providing essential tools for simulating and understanding fermionic systems, including quantum chemistry. Dive into the world of quantum computation with OpenFermion and its comprehensive features.

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:** Tools for creating, manipulating, and representing fermionic and qubit Hamiltonians.
*   **Quantum Chemistry Focus:** Specifically tailored for simulating quantum chemistry problems.
*   **Modular Plugin Architecture:** Extensible with plugins for circuit compilation, high-performance simulation, and integration with leading electronic structure packages.
*   **Cross-Platform Compatibility:** Tested on Mac, Windows, and Linux, offering flexibility for various development environments.
*   **Extensive Documentation:** Comprehensive documentation, tutorials, and API reference available.

**Get Started:**

*   **Interactive Tutorials:** Explore OpenFermion through interactive Jupyter Notebooks on |Colab|_ or |MyBinder|_.

    .. |Colab| replace:: Colab
    .. _Colab: https://colab.research.google.com/github/quantumlib/OpenFermion

    .. |MyBinder| replace:: MyBinder
    .. _MyBinder:  https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples

*   **Official Website:** Access detailed information and resources at `quantumai.google/openfermion <https://quantumai.google/openfermion>`__.

**Installation**

Install the latest stable release using `pip`:

```bash
python -m pip install --user openfermion
```

For development, refer to the original [GitHub repository](https://github.com/quantumlib/OpenFermion) for developer installation instructions.

**Plugins for Expanded Functionality**

OpenFermion's modular design allows for integration with specialized plugins, enhancing its capabilities:

*   **High-Performance Simulators:**

    *   `OpenFermion-FQE <https://github.com/quantumlib/OpenFermion-FQE>`__: Efficiently emulates fermionic quantum evolutions.

*   **Circuit Compilation:**

    *   `Forest-OpenFermion <https://github.com/rigetticomputing/forestopenfermion>`__: Integrates with Rigetti's Forest quantum computing platform.
    *   `SFOpenBoson <https://github.com/XanaduAI/SFOpenBoson>`__:  Integrates with Xanadu's Strawberry Fields for bosonic simulations.

*   **Electronic Structure Packages:**

    *   `OpenFermion-Psi4 <http://github.com/quantumlib/OpenFermion-Psi4>`__:  Integrates with Psi4 for quantum chemistry calculations.
    *   `OpenFermion-PySCF <http://github.com/quantumlib/OpenFermion-PySCF>`__:  Integrates with PySCF for quantum chemistry calculations.
    *   `OpenFermion-Dirac <https://github.com/bsenjean/Openfermion-Dirac>`__:  Integrates with DIRAC.
    *   `OpenFermion-QChem <https://github.com/qchemsoftware/OpenFermion-QChem>`__:  Integrates with Q-Chem.

**Contributing**

We welcome contributions! Please review the [contribution guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md) and sign the Contributor License Agreement (CLA) before submitting pull requests.

**Authors**

[List of authors from the original README]

**How to Cite**

When using OpenFermion in your research, please cite:

    Jarrod R McClean, Nicholas C Rubin, Kevin J Sung, Ian D Kivlichan, Xavier Bonet-Monroig,
    Yudong Cao, Chengyu Dai, E Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale,
    Thomas Häner, Tarini Hardikar, Vojtěch Havlíček, Oscar Higgott, Cupjin Huang, Josh Izaac,
    Zhang Jiang, Xinle Liu, Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman,
    Isil Ozfidan, Maxwell D Radin, Jhonathan Romero, Nicolas P D Sawaya, Bruno Senjean,
    Kanav Setia, Sukin Sim, Damian S Steiger, Mark Steudtner, Qiming Sun, Wei Sun, Daochen Wang,
    Fang Zhang, and Ryan Babbush
    *OpenFermion: The Electronic Structure Package for Quantum Computers*.
    `Quantum Science and Technology 5.3 (2020): 034014 <https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta>`__.

**Disclaimer**

Copyright 2017 The OpenFermion Developers.  This is not an official Google product.

**[View the original repository on GitHub](https://github.com/quantumlib/OpenFermion)**