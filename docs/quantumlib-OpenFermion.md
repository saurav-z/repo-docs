<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
  <br>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+ compatibility">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="Apache 2.0 License">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Release">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads">
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads">
  </a>
</div>

## OpenFermion: Simulate Fermionic Systems for Quantum Algorithms

OpenFermion is an open-source library designed to help you simulate fermionic systems and quantum chemistry algorithms, providing powerful tools for manipulating and analyzing fermionic and qubit Hamiltonians.  **[Explore the OpenFermion Repository](https://github.com/quantumlib/OpenFermion)**

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:** Provides data structures and tools for representing, manipulating, and analyzing fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Compiles and analyzes quantum algorithms for simulating fermionic systems.
*   **Integration with Quantum Computing Platforms:** Offers plugins for various quantum computing platforms and electronic structure packages.
*   **Open Source and Community Driven:**  Contribute to and benefit from an active community of researchers and developers.
*   **Comprehensive Documentation:**  Detailed documentation, tutorials, and API references are available to get you started.

**Get Started Quickly:**

You can run interactive Jupyter Notebooks using [Colab](https://colab.research.google.com/github/quantumlib/OpenFermion) or [MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples).

**Installation**

1.  **Using pip:**

    ```bash
    python -m pip install --user openfermion
    ```
2.  **Developer Install (for latest version):**
    ```bash
    git clone https://github.com/quantumlib/OpenFermion
    cd OpenFermion
    python -m pip install -e .
    ```

**Documentation**

*   [Installation](https://quantumai.google/openfermion/install)
*   [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Plugins**

OpenFermion's functionality is extended through plugins:

*   **High-Performance Simulators:** [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)
*   **Circuit Compilation:** [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion), [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson)
*   **Electronic Structure Packages:** [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4), [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF), [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac), [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem)

**How to Contribute**

We welcome contributions! Please review the [contribution guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md), sign a CLA (Contributor License Agreement), and submit pull requests with comprehensive tests and documentation.

**Authors**

(List of authors from the original README)

**How to Cite**

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

**Disclaimer**

Copyright 2017 The OpenFermion Developers. This is not an official Google product.