<!-- OpenFermion Logo -->
<p align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</p>

<!-- Badges -->
<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License: Apache 2.0">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads">
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="Downloads">
  </a>
</p>

<br>

## OpenFermion: Your Gateway to Quantum Chemistry and Fermionic System Simulation

OpenFermion is an open-source library designed to simulate fermionic systems, enabling researchers to compile and analyze quantum algorithms for quantum chemistry and beyond. [Explore the OpenFermion repository](https://github.com/quantumlib/OpenFermion).

### Key Features

*   **Fermionic Hamiltonian Manipulation:** Provides data structures and tools to represent and manipulate fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Facilitates the compilation and analysis of quantum algorithms for simulating fermionic systems.
*   **Integration with Quantum Chemistry Packages:** Offers plugins for seamless integration with popular electronic structure packages like Psi4 and PySCF.
*   **High-Performance Simulators:** Includes OpenFermion-FQE for efficient simulation of fermionic quantum evolutions.
*   **Extensive Documentation and Tutorials:** Comprehensive documentation and tutorials are available to guide users.

### Getting Started

You can run interactive Jupyter Notebooks in [Google Colab](https://colab.research.google.com/github/quantumlib/OpenFermion) or [MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples).

#### Installation

Install the latest stable release using pip:

```bash
pip install openfermion
```

or to install the latest version in development mode:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

#### Documentation

Access detailed documentation, including installation guides, API references, and tutorials:

*   **Website:** [quantumai.google/openfermion](https://quantumai.google/openfermion)
*   **Installation:** [quantumai.google/openfermion/install](https://quantumai.google/openfermion/install)
*   **API Reference:** [quantumai.google/reference/python/openfermion/all_symbols](https://quantumai.google/reference/python/openfermion/all_symbols)
*   **Tutorials:** [quantumai.google/openfermion/tutorials/intro_to_openfermion](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

#### Plugins

Extend OpenFermion's functionality with modular plugins:

*   **High-Performance Simulators:** [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)
*   **Circuit Compilation:** [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion), [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson)
*   **Electronic Structure Packages:** [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4), [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF), [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac), [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem)

### Contributing

We welcome contributions!  Please see the [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information.

### Authors

[List of Authors](Authors of OpenFermion)

### How to Cite

When using OpenFermion for research, please cite the following publication:

```
Jarrod R McClean, et al.
*OpenFermion: The Electronic Structure Package for Quantum Computers*.
Quantum Science and Technology 5.3 (2020): 034014
(https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta)
```

### Disclaimer

This is not an official Google product.