<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

<div align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python Compatibility"></a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License"></a>
  <a href="https://pypi.org/project/OpenFermion"><img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Version"></a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads"></a>
</div>

## OpenFermion: The Open-Source Toolkit for Quantum Simulation of Fermionic Systems

**OpenFermion** is an open-source Python library empowering researchers to simulate fermionic systems, including quantum chemistry, with quantum algorithms.

**Key Features:**

*   **Fermionic and Qubit Hamiltonian Manipulation:** Efficient data structures and tools for handling and manipulating Hamiltonian representations.
*   **Quantum Algorithm Compilation:**  Supports compilation and analysis of quantum algorithms for fermionic simulations.
*   **Integration with Quantum Computing Platforms:** Seamlessly integrates with platforms like Forest and Strawberry Fields via plugins.
*   **Electronic Structure Plugins:**  Integrates with popular electronic structure packages like Psi4, PySCF, and Q-Chem.
*   **Open Source and Community Driven:**  Benefit from community contributions and collaborative development.

**Get Started:**

*   **[Original Repository](https://github.com/quantumlib/OpenFermion)**
*   **Documentation:** [quantumai.google/openfermion](https://quantumai.google/openfermion)

**Installation**

Install the latest **stable** release using pip:

```bash
python -m pip install --user openfermion
```

For developers, install the latest version in development mode:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

**Plugins**

OpenFermion's modular architecture relies on plugins for extended functionality:

*   **High-Performance Simulators:**
    *   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE) - For efficient fermionic quantum evolution simulations.
*   **Circuit Compilation:**
    *   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion) - Integration with Rigetti's Forest.
    *   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson) - Integration with Strawberry Fields.
*   **Electronic Structure:**
    *   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4) - Integration with Psi4.
    *   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF) - Integration with PySCF.
    *   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac) - Integration with DIRAC.
    *   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem) - Integration with Q-Chem.

**Contribute**

We welcome contributions!  Please review our [contribution guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md) and ensure your contributions are accompanied by a Contributor License Agreement (CLA).

**How to Cite**

If you use OpenFermion in your research, please cite the following paper:

```
Jarrod R McClean, et al. *OpenFermion: The Electronic Structure Package for Quantum Computers*. Quantum Science and Technology 5.3 (2020): 034014.
```

**Disclaimer**

Copyright 2017 The OpenFermion Developers.  This is not an official Google product.