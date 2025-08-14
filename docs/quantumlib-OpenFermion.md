<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/OpenFermion/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c)](https://pypi.org/project/OpenFermion)
[![PyPI Downloads](https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads)](https://img.shields.io/pypi/dm/OpenFermion)

**OpenFermion: Your Gateway to Simulating Fermionic Systems for Quantum Chemistry and More.**  

OpenFermion is an open-source library designed to facilitate the compilation, analysis, and simulation of quantum algorithms, with a strong emphasis on fermionic systems, particularly in quantum chemistry. This library provides essential tools and data structures for manipulating and representing fermionic and qubit Hamiltonians.

[View the original repository on GitHub](https://github.com/quantumlib/OpenFermion)

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:** Efficiently create, manipulate, and analyze representations of fermionic Hamiltonians.
*   **Quantum Algorithm Compilation:** Compiles and analyzes quantum algorithms for simulating fermionic systems.
*   **Integration with Quantum Computing Platforms:** Seamlessly integrates with various quantum computing platforms, providing essential tools for researchers and developers.
*   **Electronic Structure Capabilities:** Offers tools for performing classical electronic structure calculations and integrating with popular software packages.
*   **Modular Plugin Architecture:** Leverages plugins for key functionalities, like circuit compilation and high-performance simulations.

**Get Started:**

*   **Interactive Examples:** Explore interactive Jupyter Notebooks using [Google Colab](https://colab.research.google.com/github/quantumlib/OpenFermion) or [MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples).

**Installation and Documentation**

To install the latest **stable** version of OpenFermion, ensure you have an up-to-date version of `pip <https://pip.pypa.io>`__.

*   **Documentation:** [quantumai.google/openfermion](https://quantumai.google/openfermion)

    *   [Installation](https://quantumai.google/openfermion/install)
    *   [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
    *   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

**Supported Platforms:**

OpenFermion is tested on Mac, Windows, and Linux. While electronic structure plugins are primarily compatible with Mac and Linux, a Docker image is available for cross-platform use, including Windows.  The Docker image contains a pre-installed virtual environment for easy setup. Installation and usage instructions are available in the `docker folder <https://github.com/quantumlib/OpenFermion/tree/master/docker>`__.

**Installation Options:**

*   **Developer Install:**

    ```bash
    git clone https://github.com/quantumlib/OpenFermion
    cd OpenFermion
    python -m pip install -e .
    ```

*   **Library Install:**

    ```bash
    python -m pip install --user openfermion
    ```

**Plugins**

OpenFermion's modular plugin system extends its functionality.

*   **High-performance simulators:**

    *   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): High-performance emulator for fermionic quantum evolutions.

*   **Circuit compilation plugins:**

    *   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): Integrates with Forest.
    *   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Integrates with Strawberry Fields.

*   **Electronic structure package plugins:**

    *   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Integrates with Psi4.
    *   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Integrates with PySCF.
    *   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Integrates with DIRAC.
    *   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Integrates with Q-Chem.

**Contributing**

Contributions are welcome! Please review the [contribution guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md).

*   **Contributor License Agreement (CLA):** Contributions require a CLA.
*   **Pull Requests:** Submit contributions via GitHub pull requests.
*   **Testing:** Ensure all new code includes comprehensive tests.
*   **Style Guide:** Adhere to PEP 8 guidelines.
*   **Documentation:**  Include documentation with your code.

**Support & Feedback:**

*   **Issues:** Use [GitHub Issues](https://github.com/quantumlib/OpenFermion/issues) for bug reports and feature requests.
*   **Questions:** Post questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/) with the 'openfermion' tag.

**Authors**

(List of authors - as in the original README)

**How to Cite**

Please cite the following paper when using OpenFermion for research:

(Citation information - as in the original README)

**Disclaimer**

(Disclaimer - as in the original README)