<!-- OpenFermion Logo -->
<p align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</p>

<!-- Badges -->
<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python Compatibility">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Release">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" >
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads">
  </a>
</p>

## OpenFermion: Your Gateway to Quantum Chemistry and Fermionic System Simulation

OpenFermion is an open-source library designed for compiling and analyzing quantum algorithms, providing essential tools for simulating fermionic systems, including quantum chemistry, with the goal of accelerating scientific discovery.

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:** Data structures and tools for representing and manipulating fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Facilitates the compilation of quantum algorithms for simulating fermionic systems.
*   **Integration with Quantum Chemistry Packages:** Plugins for integration with popular electronic structure packages.
*   **High-Performance Simulators:** Includes the OpenFermion-FQE for efficient simulation of fermionic quantum evolutions.

**Get Started:**

*   **Interactive Tutorials:** Explore interactive Jupyter Notebooks on |Colab|_ or |MyBinder|_.

    *   |Colab|: [Colab Link](https://colab.research.google.com/github/quantumlib/OpenFermion)
    *   |MyBinder|: [MyBinder Link](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples)

### Installation and Documentation

**Prerequisites:** Python 3.10+ and `pip`.

1.  **Install using pip:**

    ```bash
    python -m pip install --user openfermion
    ```

    or to install the latest development version

    ```bash
    git clone https://github.com/quantumlib/OpenFermion
    cd OpenFermion
    python -m pip install -e .
    ```

2.  **Documentation:** Access comprehensive documentation at: [OpenFermion Documentation](https://quantumai.google/openfermion)
    *   [Installation Guide](https://quantumai.google/openfermion/install)
    *   [API Reference](https://quantumai.google/reference/python/openfermion/all_symbols)
    *   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Supported Platforms:**

OpenFermion is tested on Mac, Windows, and Linux. Electronic structure plugins are primarily compatible with Mac and Linux. A Docker image is provided for use on other systems.  See the [Docker Instructions](https://github.com/quantumlib/OpenFermion/tree/master/docker)

### Plugins

OpenFermion's functionality is extended through modular plugin libraries, allowing for integration with various quantum computing and electronic structure tools.

**High-Performance Simulators:**

*   **OpenFermion-FQE:** [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)

**Circuit Compilation Plugins:**

*   **Forest-OpenFermion:** [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion)
*   **SFOpenBoson:** [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson)

**Electronic Structure Package Plugins:**

*   **OpenFermion-Psi4:** [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4)
*   **OpenFermion-PySCF:** [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF)
*   **OpenFermion-Dirac:** [Openfermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac)
*   **OpenFermion-QChem:** [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem)

### Contributing

We welcome contributions! Please review the [contribution guidelines](https://github.com/quantumlib/OpenFermion) and ensure your contributions are accompanied by a Contributor License Agreement (CLA).

### Get Help

*   **Issues:** Report bugs and feature requests using [GitHub Issues](https://github.com/quantumlib/OpenFermion/issues).
*   **Questions:** Ask questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/) with the tag 'openfermion'.

### Authors

A list of authors can be found in the original [README](https://github.com/quantumlib/OpenFermion).

### How to Cite

If you use OpenFermion in your research, please cite the following paper:

```
Jarrod R McClean, Nicholas C Rubin, Kevin J Sung, Ian D Kivlichan, Xavier Bonet-Monroig,
Yudong Cao, Chengyu Dai, E Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale,
Thomas Häner, Tarini Hardikar, Vojtěch Havlíček, Oscar Higgott, Cupjin Huang, Josh Izaac,
Zhang Jiang, Xinle Liu, Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman,
Isil Ozfidan, Maxwell D Radin, Jhonathan Romero, Nicolas P D Sawaya, Bruno Senjean,
Kanav Setia, Sukin Sim, Damian S Steiger, Mark Steudtner, Qiming Sun, Wei Sun, Daochen Wang,
Fang Zhang, and Ryan Babbush
*OpenFermion: The Electronic Structure Package for Quantum Computers*.
Quantum Science and Technology 5.3 (2020): 034014 <https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta>.
```

### Disclaimer

Copyright 2017 The OpenFermion Developers. This is not an official Google product.