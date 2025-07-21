<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/OpenFermion/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c)](https://pypi.org/project/OpenFermion)
[![Downloads](https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads)](https://img.shields.io/pypi/dm/OpenFermion)

# OpenFermion: Your Gateway to Quantum Chemistry and Fermionic System Simulation

OpenFermion is a powerful open-source library designed for simulating fermionic systems and quantum chemistry problems on quantum computers.

**[Explore the OpenFermion Repository](https://github.com/quantumlib/OpenFermion)**

## Key Features

*   **Fermionic and Qubit Hamiltonian Manipulation:** Provides data structures and tools to obtain, manipulate, and analyze fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Facilitates the compilation and analysis of quantum algorithms for simulating fermionic systems.
*   **Integration with Quantum Computing Platforms:** Integrates with various quantum computing platforms through plugins.
*   **Extensive Plugin Ecosystem:** Supports integration with various electronic structure packages and quantum circuit compilation tools, including:
    *   OpenFermion-FQE
    *   Forest-OpenFermion
    *   SFOpenBoson
    *   OpenFermion-Psi4
    *   OpenFermion-PySCF
    *   OpenFermion-Dirac
    *   OpenFermion-QChem
*   **User-Friendly Documentation:** Comprehensive documentation with installation guides, API references, and tutorials.

## Installation and Usage

### Prerequisites

*   Python 3.10 or higher
*   `pip` (Python package installer)

### Installation Options

*   **Stable Release (using pip):**
    ```bash
    python -m pip install --user openfermion
    ```
*   **Development Version (from source):**
    ```bash
    git clone https://github.com/quantumlib/OpenFermion
    cd OpenFermion
    python -m pip install -e .
    ```

### Documentation

*   [Documentation](https://quantumai.google/openfermion)
*   [Installation Guide](https://quantumai.google/openfermion/install)
*   [API Reference](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

### Docker Installation (for Windows/other platforms)

A Docker image is available for users who may have difficulty installing OpenFermion or its plugins on their native operating system. The Docker image provides a virtual environment with OpenFermion and select plugins pre-installed. See the [docker folder](https://github.com/quantumlib/OpenFermion/tree/master/docker) for more information.

## Contributing

We welcome contributions! Please refer to the [How to Contribute](#how-to-contribute) section in the original README (link at the top) for more details on guidelines and contribution process.

## Authors

OpenFermion is developed by a team of researchers and engineers.  See the original README (link at the top) for a complete list of authors.

## How to cite

When using OpenFermion for research projects, please cite:

    Jarrod R McClean, Nicholas C Rubin, Kevin J Sung, Ian D Kivlichan, Xavier Bonet-Monroig,
    Yudong Cao, Chengyu Dai, E Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale,
    Thomas Häner, Tarini Hardikar, Vojtĕch Havlíček, Oscar Higgott, Cupjin Huang, Josh Izaac,
    Zhang Jiang, Xinle Liu, Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman,
    Isil Ozfidan, Maxwell D Radin, Jhonathan Romero, Nicolas P D Sawaya, Bruno Senjean,
    Kanav Setia, Sukin Sim, Damian S Steiger, Mark Steudtner, Qiming Sun, Wei Sun, Daochen Wang,
    Fang Zhang, and Ryan Babbush
    *OpenFermion: The Electronic Structure Package for Quantum Computers*.
    `Quantum Science and Technology 5.3 (2020): 034014 <https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta>`__.