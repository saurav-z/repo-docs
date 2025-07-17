<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
  <br/>
  <br/>
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

## OpenFermion: Simulate and Analyze Fermionic Systems for Quantum Algorithms

**OpenFermion** is an open-source library providing essential tools and data structures to simulate and analyze quantum algorithms, particularly for quantum chemistry, empowering researchers to explore the potential of quantum computing for complex scientific challenges. Check out the original repo [here](https://github.com/quantumlib/OpenFermion).

**Key Features:**

*   **Fermionic and Qubit Hamiltonian Representation:** Provides data structures and tools for manipulating fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Facilitates the compilation and analysis of quantum algorithms.
*   **Integration with Plugins:** Leverages modular plugins for circuit simulation, electronic structure calculations, and more.
*   **Open Source and Community-Driven:** Fosters collaboration and innovation in the field of quantum computing.
*   **Comprehensive Documentation:** Offers detailed API documentation and tutorials.

You can run the interactive Jupyter Notebooks in [Colab](https://colab.research.google.com/github/quantumlib/OpenFermion) or [MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples).

## Installation and Documentation

### Installation

Install the latest **stable** OpenFermion using `pip <https://pip.pypa.io>`__:

```bash
python -m pip install --user openfermion
```

### Documentation

Comprehensive documentation is available at:

*   `quantumai.google/openfermion <https://quantumai.google/openfermion>`__
*   `Installation <https://quantumai.google/openfermion/install>`__
*   `API Docs <https://quantumai.google/reference/python/openfermion/all_symbols>`__
*   `Tutorials <https://quantumai.google/openfermion/tutorials/intro_to_openfermion>`__

### Developer Installation

To install the latest development version:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

## Plugins

OpenFermion utilizes modular plugin libraries for extended functionality.

### High-Performance Simulators

*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): High-performance emulator for fermionic quantum evolutions.

### Circuit Compilation Plugins

*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): Integration with Forest.
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Integration with Strawberry Fields.

### Electronic Structure Package Plugins

*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Integration with Psi4.
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Integration with PySCF.
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Integration with DIRAC.
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Integration with Q-Chem.

## Contributing

Contributions are welcome! Please review the guidelines and CLA information at:
*   [How to contribute](https://github.com/quantumlib/OpenFermion/blob/main/README.md#how-to-contribute)

## Authors

*   A complete list of authors can be found in the original README.

## How to Cite

To cite OpenFermion, please use the following reference:

```bibtex
@article{McClean2020,
  title={OpenFermion: The Electronic Structure Package for Quantum Computers},
  author={McClean, Jarrod R and Rubin, Nicholas C and Sung, Kevin J and Kivlichan, Ian D and Bonet-Monroig, Xavier and Cao, Yudong and Dai, Chengyu and Fried, E Schuyler and Gidney, Craig and Gimby, Brendan and others},
  journal={Quantum Science and Technology},
  volume={5},
  number={3},
  pages={034014},
  year={2020},
  publisher={IOP Publishing}
}
```

## Disclaimer

This is not an official Google product.