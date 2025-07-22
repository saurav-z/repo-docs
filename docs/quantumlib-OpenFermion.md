<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion logo" width="75%">
</div>

[![Python](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/OpenFermion/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c)](https://pypi.org/project/OpenFermion)
[![Downloads](https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads)](https://img.shields.io/pypi/dm/OpenFermion)

## OpenFermion: Simulate Fermionic Systems with Quantum Algorithms

OpenFermion is a powerful open-source library designed for compiling and analyzing quantum algorithms, enabling researchers to simulate and understand fermionic systems, including quantum chemistry, on quantum computers.  Explore the code on [GitHub](https://github.com/quantumlib/OpenFermion).

**Key Features:**

*   **Fermionic Hamiltonian Representation:** Provides data structures and tools for working with fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:**  Facilitates the compilation of quantum algorithms for simulation.
*   **Open Source:**  Freely available under the Apache 2.0 license.
*   **Modular Plugins:** Extensible with plugins for high-performance simulation, circuit compilation, and electronic structure calculations.
*   **Integration with Popular Tools:** Seamlessly integrates with tools like Psi4, PySCF, and more.

## Installation and Documentation

**Installation**

*   Install the latest stable release using pip:

    ```bash
    python -m pip install --user openfermion
    ```
*   For developers, install in editable mode:

    ```bash
    git clone https://github.com/quantumlib/OpenFermion
    cd OpenFermion
    python -m pip install -e .
    ```

**Documentation**

*   [Official Documentation](https://quantumai.google/openfermion)
    *   [Installation Guide](https://quantumai.google/openfermion/install)
    *   [API Reference](https://quantumai.google/reference/python/openfermion/all_symbols)
    *   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Supported Platforms:**

OpenFermion is tested on Mac, Windows, and Linux. Electronic structure plugins have compatibility on these platforms. Docker images are available for broader platform support.

## Plugins

Enhance OpenFermion's functionality with these plugins:

**High-Performance Simulators**

*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): Optimize fermionic quantum evolutions.

**Circuit Compilation Plugins**

*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): Integrate with Rigetti's Forest.
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Integrate with Xanadu's Strawberry Fields.

**Electronic Structure Package Plugins**

*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Interface with Psi4.
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Interface with PySCF.
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Interface with DIRAC.
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Interface with Q-Chem.

## How to Contribute

Contributions are welcome!  Please follow the guidelines in the documentation.

*   Sign the Contributor License Agreement (CLA) at https://cla.developers.google.com/.
*   Submit contributions via GitHub pull requests.
*   Ensure code is well-tested and follows PEP 8 style guidelines.
*   Include comprehensive documentation.

**For questions or bug reports:**

*   Use [GitHub issues](https://github.com/quantumlib/OpenFermion/issues).
*   Post questions to the Quantum Computing Stack Exchange with the 'openfermion' tag.

## Authors

[List of Authors](Original README Authors Section)

## How to Cite

When using OpenFermion, please cite the following publication:

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
```

## Disclaimer

This is not an official Google product.