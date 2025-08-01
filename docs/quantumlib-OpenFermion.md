<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
  <br>
  <br>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License: Apache 2.0">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" >
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="Downloads">
  </a>
</div>

OpenFermion is an open-source Python library that empowers researchers and developers to simulate and analyze fermionic systems, accelerating quantum computing research.  Find the original repo [here](https://github.com/quantumlib/OpenFermion).

## Key Features of OpenFermion

*   **Fermionic Hamiltonian Manipulation:** Provides data structures and tools to represent and manipulate fermionic and qubit Hamiltonians, crucial for quantum chemistry simulations.
*   **Quantum Algorithm Compilation:** Offers tools for compiling and analyzing quantum algorithms.
*   **Integration with Quantum Simulators:** Seamlessly integrates with high-performance simulators like OpenFermion-FQE, and with quantum circuit compilation plugins such as Forest-OpenFermion, SFOpenBoson.
*   **Electronic Structure Plugins:** Supports integration with leading electronic structure packages like Psi4, PySCF, and Q-Chem.
*   **Cross-Platform Compatibility:** Tested and supported on macOS, Windows, and Linux, with Docker images available for consistent environments.
*   **Extensive Documentation:** Comprehensive documentation and tutorials available to help users get started quickly.

## Installation and Documentation

**Install OpenFermion:**

```bash
pip install --user openfermion
```

**Documentation:**
*   [Installation](https://quantumai.google/openfermion/install)
*   [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

## Plugins

OpenFermion relies on modular plugin libraries for significant functionality. Plugins are used to simulate and compile quantum circuits and to perform classical electronic structure calculations.

### High-Performance Simulators

*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): A high-performance emulator of fermionic quantum evolutions.

### Circuit Compilation Plugins

*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): Integration with Forest.
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Integration with Strawberry Fields.

### Electronic Structure Package Plugins

*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Integration with Psi4.
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Integration with PySCF.
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Integration with DIRAC.
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Integration with Q-Chem.

## Contributing

Contributions are welcome!  Consult the [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more on using pull requests.  Make sure your code includes tests, adheres to PEP 8, and includes documentation.

## How to Cite

If you use OpenFermion in your research, please cite the following paper:

```bibtex
@article{mcclean2020openfermion,
  title={OpenFermion: The Electronic Structure Package for Quantum Computers},
  author={McClean, Jarrod R and Rubin, Nicholas C and Sung, Kevin J and Kivlichan, Ian D and Bonet-Monroig, Xavier and Cao, Yudong and Dai, Chengyu and Fried, E Schuyler and Gidney, Craig and Gimby, Brendan and others},
  journal={Quantum Science and Technology},
  volume={5},
  number={3},
  pages={034014},
  year={2020},
  publisher={IOP Publishing}
}