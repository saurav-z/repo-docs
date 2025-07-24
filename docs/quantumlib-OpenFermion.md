<!-- OpenFermion Logo -->
<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion logo" width="75%">
</div>

<!-- Badges -->
<div align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads">
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="Downloads">
  </a>
</div>

## OpenFermion: A Powerful Library for Quantum Chemistry and Fermionic System Simulation

OpenFermion is an open-source library designed to empower researchers and developers in the field of quantum computing with tools for simulating fermionic systems, with a strong focus on quantum chemistry.

**Key Features:**

*   **Fermionic and Qubit Hamiltonian Manipulation:** Provides data structures and tools for working with and manipulating representations of fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation and Analysis:**  Facilitates the compilation and analysis of quantum algorithms tailored for simulating fermionic systems.
*   **Integration with Quantum Computing Platforms:** Supports integration with various quantum computing platforms and software packages through plugin libraries.
*   **Modular Architecture:** Designed with modularity in mind, allowing for easy extension and integration of new features and functionalities through plugins.
*   **Comprehensive Documentation:** Offers extensive documentation, tutorials, and examples to help users get started and understand the library's capabilities.

**Get Started:**

*   Explore interactive Jupyter Notebooks on:
    *   Colab: [![Colab](https://colab.research.google.com/github/quantumlib/OpenFermion)](https://colab.research.google.com/github/quantumlib/OpenFermion)
    *   MyBinder: [![MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples)](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples)

**Installation and Documentation**

OpenFermion can be installed via pip:

```bash
python -m pip install --user openfermion
```

Refer to the official documentation at [quantumai.google/openfermion](https://quantumai.google/openfermion) for detailed installation instructions, API reference, and tutorials.

*   [Installation](https://quantumai.google/openfermion/install)
*   [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Plugin Ecosystem**

OpenFermion leverages plugins to extend its functionality:

*   **High-performance simulators:**
    *   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)

*   **Circuit compilation plugins:**
    *   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion)
    *   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson)

*   **Electronic structure package plugins:**
    *   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4)
    *   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF)
    *   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac)
    *   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem)

**Contribute**

We welcome contributions! Please read the [contributing guidelines](https://github.com/quantumlib/OpenFermion) for details.

**Authors & Citation**

For a full list of authors, please refer to the original repository. When using OpenFermion, please cite:

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

**[View the OpenFermion repository on GitHub](https://github.com/quantumlib/OpenFermion)**