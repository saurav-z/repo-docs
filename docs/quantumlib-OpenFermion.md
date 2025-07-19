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
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Version">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads">
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="Downloads">
  </a>
</div>

## OpenFermion: Simulate Quantum Chemistry and Fermionic Systems

OpenFermion is a powerful, open-source library designed for simulating fermionic systems, including quantum chemistry, by providing tools for representing and manipulating fermionic and qubit Hamiltonians.  Explore the original repository on [GitHub](https://github.com/quantumlib/OpenFermion).

**Key Features:**

*   **Fermionic System Simulation:**  Provides data structures and tools for representing and manipulating fermionic Hamiltonians.
*   **Quantum Algorithm Compilation and Analysis:**  Enables the compilation and analysis of quantum algorithms for simulating fermionic systems.
*   **Modular Plugin Architecture:**  Offers extensibility through plugins for high-performance simulation, circuit compilation, and integration with electronic structure packages.
*   **Integration with Popular Tools:** Supports integration with tools like Psi4, PySCF, and more, providing a comprehensive ecosystem for quantum chemistry research.

**Get Started:**

*   **Interactive Tutorials:** Explore interactive Jupyter Notebooks in [Colab](https://colab.research.google.com/github/quantumlib/OpenFermion) or [MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples).

### Installation

**Prerequisites:** Python 3.10+ and pip.

**Installation (Stable Release):**

```bash
python -m pip install --user openfermion
```

**Installation (Development Mode):**

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

### Documentation and Resources

*   **Documentation:**  Find comprehensive documentation at [quantumai.google/openfermion](https://quantumai.google/openfermion).
*   **API Reference:** Explore the API documentation at [quantumai.google/reference/python/openfermion/all_symbols](https://quantumai.google/reference/python/openfermion/all_symbols).
*   **Tutorials:**  Learn the basics with tutorials at [quantumai.google/openfermion/tutorials/intro_to_openfermion](https://quantumai.google/openfermion/tutorials/intro_to_openfermion).
*   **Docker:**  For users facing installation challenges, a Docker image is available in the [docker folder](https://github.com/quantumlib/OpenFermion/tree/master/docker).

### Plugins

OpenFermion leverages plugins for extended functionality:

*   **High-Performance Simulators:**
    *   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE):  For high-performance simulation of fermionic quantum evolutions.
*   **Circuit Compilation Plugins:**
    *   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion):  Integration with Forest.
    *   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson):  Integration with Strawberry Fields.
*   **Electronic Structure Package Plugins:**
    *   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Integration with Psi4.
    *   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Integration with PySCF.
    *   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Integration with DIRAC.
    *   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Integration with Q-Chem.

### Contributing

Contributions are welcome! Please refer to the project's contribution guidelines and ensure you have signed a Contributor License Agreement (CLA).

*   **Issue Tracking:** Use [GitHub issues](https://github.com/quantumlib/OpenFermion/issues) for bug reports and feature requests.
*   **Questions:**  Ask questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/) with the tag "openfermion".

### Authors

(List of authors from the original README)

### Citation

If you use OpenFermion in your research, please cite the following paper:

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

### Disclaimer

This is not an official Google product.