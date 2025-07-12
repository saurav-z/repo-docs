<!-- OpenFermion Logo -->
<p align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License: Apache 2.0" />
  <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI" />
  <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="Downloads" />
</p>

# OpenFermion: Quantum Computing for Fermionic Systems

OpenFermion is an open-source library enabling researchers to explore and simulate fermionic systems, including quantum chemistry, with tools for building, manipulating, and analyzing qubit and fermionic Hamiltonians.  [Explore the OpenFermion repository](https://github.com/quantumlib/OpenFermion).

## Key Features

*   **Fermionic Hamiltonian Construction:** Build and represent fermionic Hamiltonians efficiently.
*   **Qubitization:** Convert fermionic Hamiltonians into qubit representations suitable for quantum computers.
*   **Circuit Compilation:** Compile and optimize quantum circuits for simulating fermionic systems.
*   **Electronic Structure Plugins:** Integrate with popular electronic structure packages for classical calculations and data.
*   **High-Performance Simulators:** Utilize advanced simulation tools for fermionic quantum evolution.
*   **Modular Design:**  Easily extendable with plugins for various quantum computing frameworks.

## Installation and Documentation

Get started with OpenFermion by following these installation steps.  Comprehensive documentation is available to guide you through the library's functionalities.

### Installation

Install the latest stable version of OpenFermion using pip:

```bash
python -m pip install --user openfermion
```

### Documentation

Find detailed documentation and tutorials at:

*   [Installation Guide](https://quantumai.google/openfermion/install)
*   [API Documentation](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

### Developer Installation

For the latest development version:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

## Plugins

OpenFermion leverages plugins for expanded functionality.

### High-Performance Simulators

*   **OpenFermion-FQE:** High-performance fermionic quantum evolution emulator.

### Circuit Compilation Plugins

*   **Forest-OpenFermion:** Integration with Forest quantum computing platform.
*   **SFOpenBoson:** Integration with Strawberry Fields, a quantum computing platform.

### Electronic Structure Package Plugins

*   **OpenFermion-Psi4:**  Integration with Psi4 for electronic structure calculations.
*   **OpenFermion-PySCF:** Integration with PySCF for electronic structure calculations.
*   **OpenFermion-Dirac:** Integration with DIRAC.
*   **OpenFermion-QChem:** Integration with Q-Chem.

## Contributing

Contributions to OpenFermion are welcome!  Please review the [Contribution Guidelines](https://github.com/quantumlib/OpenFermion/blob/master/CONTRIBUTING.md) for details on how to contribute, including the Contributor License Agreement (CLA) requirements and the code style.

### Resources:
*   [GitHub Issues](https://github.com/quantumlib/OpenFermion/issues)
*   [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/) (tag: openfermion)

## Authors

OpenFermion is developed by a team of researchers from Google, universities, and research institutions.  [See full list of authors in the original README](https://github.com/quantumlib/OpenFermion).

## Citation

If you use OpenFermion in your research, please cite:

```
Jarrod R McClean, Nicholas C Rubin, Kevin J Sung, Ian D Kivlichan, Xavier Bonet-Monroig,
Yudong Cao, Chengyu Dai, E Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale,
Thomas Häner, Tarini Hardikar, Vojtěch Havlíček, Oscar Higgott, Cupjin Huang, Josh Izaac,
Zhang Jiang, Xinle Liu, Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman,
Isil Ozfidan, Maxwell D Radin, Jhonathan Romero, Nicolas P D Sawaya, Bruno Senjean,
Kanav Setia, Sukin Sim, Damian S Steiger, Mark Steudtner, Qiming Sun, Wei Sun, Daochen Wang,
Fang Zhang, and Ryan Babbush
*OpenFermion: The Electronic Structure Package for Quantum Computers*.
`Quantum Science and Technology 5.3 (2020): 034014 <https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta>`__.
```

## Disclaimer

This is not an official Google product.