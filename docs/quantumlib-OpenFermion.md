<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
  <br>
  <br>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python Compatibility">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Release">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads">
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads">
</div>

OpenFermion is an open-source library enabling researchers to simulate fermionic systems and quantum chemistry problems, paving the way for breakthroughs in quantum computing.

**Key Features:**

*   **Fermionic and Qubit Hamiltonian Manipulation:** Provides data structures and tools for representing and manipulating fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation and Analysis:**  Facilitates the compilation and analysis of quantum algorithms for simulating fermionic systems.
*   **Integration with Quantum Chemistry Packages:**  Offers plugins for seamless integration with popular electronic structure packages (Psi4, PySCF, Q-Chem, DIRAC).
*   **High-Performance Simulators:**  Includes `OpenFermion-FQE`, a high-performance emulator for fermionic quantum evolutions.
*   **Circuit Compilation:** Supports integration with quantum circuit compilation tools like Forest and Strawberry Fields via plugins.
*   **Cross-Platform Compatibility:** Tested on Mac, Windows, and Linux. Docker image available for consistent environment setup.

**Installation and Documentation**

To get started with OpenFermion, follow these steps:

1.  **Install:** Install the latest stable release using pip:
    ```bash
    python -m pip install --user openfermion
    ```
2.  **Developer Install (for latest version):**
    ```bash
    git clone https://github.com/quantumlib/OpenFermion
    cd OpenFermion
    python -m pip install -e .
    ```

**Documentation:**  Access comprehensive documentation and resources:

*   [Official Website](https://quantumai.google/openfermion)
*   [Installation Guide](https://quantumai.google/openfermion/install)
*   [API Documentation](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Plugins**

OpenFermion's extensibility is facilitated by plugins for advanced functionality:

*   **OpenFermion-FQE:** High-performance fermionic quantum evolution emulator ([link](https://github.com/quantumlib/OpenFermion-FQE))
*   **Circuit Compilation Plugins:**
    *   Forest-OpenFermion ([link](https://github.com/rigetticomputing/forestopenfermion))
    *   SFOpenBoson ([link](https://github.com/XanaduAI/SFOpenBoson))
*   **Electronic Structure Plugins:**
    *   OpenFermion-Psi4 ([link](http://github.com/quantumlib/OpenFermion-Psi4))
    *   OpenFermion-PySCF ([link](http://github.com/quantumlib/OpenFermion-PySCF))
    *   OpenFermion-Dirac ([link](https://github.com/bsenjean/Openfermion-Dirac))
    *   OpenFermion-QChem ([link](https://github.com/qchemsoftware/OpenFermion-QChem))

**Contributing**

We welcome contributions! Please see the [GitHub repository](https://github.com/quantumlib/OpenFermion) for contribution guidelines, including the Contributor License Agreement (CLA), code style (PEP 8), and testing procedures.

**Authors**

(List of authors from the original README)

**How to Cite**

If you use OpenFermion in your research, please cite the following paper:

*   Jarrod R McClean, et al. *OpenFermion: The Electronic Structure Package for Quantum Computers*. Quantum Science and Technology 5.3 (2020): 034014 ([https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta](https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta))

**Disclaimer**

Copyright 2017 The OpenFermion Developers.  This is not an official Google product.