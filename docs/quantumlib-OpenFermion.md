<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion logo" width="75%">
</div>

<div align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads">
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="Downloads">
  </a>
</div>

## OpenFermion: Accelerate Your Quantum Chemistry and Fermionic Simulations

OpenFermion is an open-source Python library designed for the development and analysis of quantum algorithms, specifically focusing on simulating fermionic systems and quantum chemistry.

**Key Features:**

*   **Fermionic and Qubit Hamiltonian Manipulation:** Provides data structures and tools to represent and manipulate fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Enables the compilation and analysis of quantum algorithms.
*   **Plugin Ecosystem:** Offers modular plugins for circuit compilation, simulation, and electronic structure calculations, including integrations with:
    *   OpenFermion-FQE
    *   Forest-OpenFermion
    *   SFOpenBoson
    *   OpenFermion-Psi4
    *   OpenFermion-PySCF
    *   OpenFermion-Dirac
    *   OpenFermion-QChem
*   **Cross-Platform Compatibility:** Tested on Mac, Windows, and Linux, with a Docker image for simplified setup.

**Get Started:**

*   **Documentation:** Explore the comprehensive documentation at [quantumai.google/openfermion](https://quantumai.google/openfermion).
    *   [Installation](https://quantumai.google/openfermion/install)
    *   [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
    *   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)
*   **Interactive Examples:** Run interactive Jupyter Notebooks in [Colab](https://colab.research.google.com/github/quantumlib/OpenFermion) or [MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples).
*   **Installation:**
    *   **Stable Release (using pip):** `python -m pip install --user openfermion`
    *   **Development Mode (for contributing):**
        ```bash
        git clone https://github.com/quantumlib/OpenFermion
        cd OpenFermion
        python -m pip install -e .
        ```

**How to Contribute:**

We welcome contributions! Please review our [contributing guidelines](https://github.com/quantumlib/OpenFermion#how-to-contribute) for more information.

**Repository:** [https://github.com/quantumlib/OpenFermion](https://github.com/quantumlib/OpenFermion)

**Authors:**

[List of Authors - see original README for list]

**How to Cite:**

When using OpenFermion in your research, please cite the following paper:

    Jarrod R McClean, et al.
    *OpenFermion: The Electronic Structure Package for Quantum Computers*.
    `Quantum Science and Technology 5.3 (2020): 034014 <https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta>`__.