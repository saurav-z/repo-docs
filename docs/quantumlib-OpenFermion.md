<!-- OpenFermion Logo -->
<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

<!-- Badges -->
<div align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python Compatibility">
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

<!-- Vertical Space -->
<br>
<br>

## OpenFermion: Your Gateway to Quantum Chemistry and Fermionic System Simulation

OpenFermion is an open-source library designed to simulate fermionic systems on quantum computers, offering tools for compiling and analyzing quantum algorithms, making it a powerful resource for quantum chemistry and beyond. [Explore the original repository](https://github.com/quantumlib/OpenFermion).

**Key Features:**

*   **Data Structures:** Provides data structures for representing fermionic and qubit Hamiltonians.
*   **Algorithm Compilation:** Compiles and analyzes quantum algorithms for simulating fermionic systems.
*   **Quantum Chemistry Focus:** Specifically designed to aid in quantum chemistry simulations.
*   **Modular Plugins:** Leverages plugins for simulators, circuit compilation, and electronic structure calculations.
*   **Integration:** Integrates with popular platforms like Colab, MyBinder and other tools.

**Interactive Notebooks:**
Explore OpenFermion through interactive Jupyter Notebooks:

*   [Colab](https://colab.research.google.com/github/quantumlib/OpenFermion)
*   [MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples)

## Installation and Documentation

### Installation

Install the latest stable version of OpenFermion using `pip`:

```bash
python -m pip install --user openfermion
```

For development, clone the repo and install in editable mode:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

### Documentation

Comprehensive documentation is available at:

*   **Main Documentation:**  [https://quantumai.google/openfermion](https://quantumai.google/openfermion)
*   **Installation:** [https://quantumai.google/openfermion/install](https://quantumai.google/openfermion/install)
*   **API Reference:** [https://quantumai.google/reference/python/openfermion/all_symbols](https://quantumai.google/reference/python/openfermion/all_symbols)
*   **Tutorials:** [https://quantumai.google/openfermion/tutorials/intro_to_openfermion](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Operating System Support:**  OpenFermion is tested on Mac, Windows, and Linux. Electronic structure plugins have the best compatibility on Mac or Linux. A Docker image is available for Windows and other systems via the `docker folder <https://github.com/quantumlib/OpenFermion/tree/master/docker>`__.

## Plugins

OpenFermion's functionality is extended through modular plugins:

### High-Performance Simulators
*   **OpenFermion-FQE:**  A high-performance emulator for fermionic quantum evolutions.  [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)

### Circuit Compilation Plugins
*   **Forest-OpenFermion:** Integration with `Forest <https://www.rigetti.com/forest>`.
*   **SFOpenBoson:** Integration with `Strawberry Fields <https://github.com/XanaduAI/strawberryfields>`.

### Electronic Structure Package Plugins
*   **OpenFermion-Psi4:** Integration with `Psi4 <http://psicode.org>`.
*   **OpenFermion-PySCF:** Integration with `PySCF <https://github.com/sunqm/pyscf>`.
*   **OpenFermion-Dirac:** Integration with `DIRAC <http://diracprogram.org/doku.php>`.
*   **OpenFermion-QChem:** Integration with `Q-Chem <https://www.q-chem.com>`.

## Contributing

Contributions are welcome!  Please follow the guidelines outlined in the original repository.
Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests. Make sure your code includes thorough tests and adheres to the project's style guide (PEP 8).

*   **CLA:** Contributions require a Contributor License Agreement (CLA) from Google.
*   **Pull Requests:**  Submit contributions via GitHub pull requests.
*   **Testing:** Ensure your code includes extensive tests.
*   **Style Guide:**  Follow PEP 8 guidelines.
*   **Documentation:** Include documentation for any new code.

## Support

*   **Issues:**  Report bugs and request features via [Github issues](https://github.com/quantumlib/OpenFermion/issues).
*   **Questions:** Post questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/) with the tag 'openfermion'.

## Authors

A comprehensive list of authors is provided in the original README.

## How to Cite

When using OpenFermion for research, please cite the following publication:

```
Jarrod R McClean, et al. *OpenFermion: The Electronic Structure Package for Quantum Computers*. Quantum Science and Technology 5.3 (2020): 034014. [https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta](https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta)
```

## Disclaimer

Copyright 2017 The OpenFermion Developers. This is not an official Google product.