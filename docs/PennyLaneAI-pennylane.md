# PennyLane: The Open-Source Quantum Computing Library

**Unlock the power of quantum computing and accelerate your research with PennyLane, a versatile Python library for quantum machine learning, quantum chemistry, and more.** ([View on GitHub](https://github.com/PennyLaneAI/pennylane))

<p align="center">
  <!-- Tests (GitHub actions) -->
  <a href="https://github.com/PennyLaneAI/pennylane/actions?query=workflow%3ATests">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/PennyLane/tests.yml?branch=master&style=flat-square" alt="Tests Status"/>
  </a>
  <!-- CodeCov -->
  <a href="https://codecov.io/gh/PennyLaneAI/pennylane">
    <img src="https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane/master.svg?logo=codecov&style=flat-square" alt="Code Coverage"/>
  </a>
  <!-- ReadTheDocs -->
  <a href="https://docs.pennylane.ai/en/latest">
    <img src="https://readthedocs.com/projects/xanaduai-pennylane/badge/?version=latest&style=flat-square" alt="Documentation"/>
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/PennyLane">
    <img src="https://img.shields.io/pypi/v/PennyLane.svg?style=flat-square" alt="PyPI Version"/>
  </a>
  <!-- Forum -->
  <a href="https://discuss.pennylane.ai">
    <img src="https://img.shields.io/discourse/https/discuss.pennylane.ai/posts.svg?logo=discourse&style=flat-square" alt="Discussion Forum"/>
  </a>
  <!-- License -->
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/pypi/l/PennyLane.svg?logo=apache&style=flat-square" alt="License"/>
  </a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/pl-logo-lightmode.png#gh-light-mode-only" width="700px" alt="PennyLane Logo Light Mode">
  <img src="./doc/_static/readme/pl-logo-darkmode.png#gh-dark-mode-only" width="700px" onerror="this.style.display='none'" alt="PennyLane Logo Dark Mode"/>
</p>

## Key Features

PennyLane empowers researchers and developers with a comprehensive set of tools for quantum computation and its applications.

*   **Program Quantum Computers:** Build quantum circuits using a wide array of gates and measurements, and run them on high-performance simulators or various quantum hardware devices.  Take advantage of advanced features like mid-circuit measurements and error mitigation.
*   **Master Quantum Algorithms:** Explore a wide range of quantum algorithms, from NISQ to fault-tolerant models. Analyze performance, visualize circuits, and access tools for quantum chemistry and algorithm development.
*   **Quantum Machine Learning:** Seamlessly integrate with frameworks like PyTorch, TensorFlow, JAX, Keras, and NumPy to define and train hybrid quantum-classical models. Utilize quantum-aware optimizers and hardware-compatible gradients for cutting-edge research in QML.
*   **Quantum Datasets:** Access pre-simulated, high-quality quantum datasets to accelerate algorithm development and reduce time-to-research. Browse available datasets or contribute your own data.
*   **Compilation and Performance:** Benefit from experimental support for just-in-time compilation. Compile your hybrid workflows for advanced features such as adaptive circuits, real-time measurement feedback, and unbounded loops. (Explore [Catalyst](https://github.com/pennylaneai/catalyst) for details.)

For more details and additional features, please see the [PennyLane website](https://pennylane.ai/features/).

## Installation

PennyLane requires Python 3.11 or higher. Install PennyLane and all dependencies using pip:

```bash
python -m pip install pennylane
```

## Docker Support

PennyLane offers Docker images for easy setup and deployment.  Find them on the [PennyLane Docker Hub page](https://hub.docker.com/u/pennylaneai). For further details, consult the [PennyLane Docker documentation](https://docs.pennylane.ai/projects/lightning/en/stable/dev/docker.html).

## Getting Started

Start your quantum journey with PennyLane using our comprehensive [quickstart guide](https://docs.pennylane.ai/en/stable/introduction/pennylane.html).

### Key Resources:

*   [Research-oriented Demos](https://pennylane.ai/qml/demonstrations)
*   [Learn Quantum Programming](https://pennylane.ai/qml/) (including the [Codebook](https://pennylane.ai/codebook/) and [Coding Challenges](https://pennylane.ai/challenges/))
*   [Frequently Asked Questions](https://pennylane.ai/faq)
*   [Glossary](https://pennylane.ai/qml/glossary)
*   [Videos](https://pennylane.ai/qml/videos)

Explore our comprehensive [documentation](https://pennylane.readthedocs.io) and developer guides.

## Demos

Discover cutting-edge quantum algorithms and explore the capabilities of PennyLane through our interactive [demos](https://pennylane.ai/qml/demonstrations).

<a href="https://pennylane.ai/qml/demonstrations">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/demos.png" width="900px" alt="PennyLane Demos">
</a>

Contribute your own demo by following our [demo submission guide](https://pennylane.ai/qml/demos_submission).

## Research Applications

PennyLane is at the forefront of research in quantum computing, quantum machine learning, and quantum chemistry. Here are some recent publications that utilize PennyLane:

*   **Quantum Computing**: [Fast quantum circuit cutting with randomized measurements](https://quantum-journal.org/papers/q-2023-03-02-934/)
*   **Quantum Machine Learning**: [Better than classical? The subtle art of benchmarking quantum machine learning models](https://arxiv.org/abs/2403.07059)
*   **Quantum Chemistry**: [Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization](https://quantum-journal.org/papers/q-2024-06-13-1371/)

Share your research needs on [GitHub](https://github.com/PennyLaneAI/pennylane/issues/new?assignees=&labels=enhancement+%3Asparkles%3A&projects=&template=feature_request.yml) or our [website](https://pennylane.ai/research) to help shape future developments.

## Contributing to PennyLane

Contribute to the project by forking the PennyLane repository and submitting a [pull request](https://help.github.com/articles/about-pull-requests/). Significant contributors are acknowledged in PennyLane releases and publications.

We welcome bug reports, new feature suggestions, and links to interesting projects.

Refer to our [contributions page](https://github.com/PennyLaneAI/pennylane/blob/master/.github/CONTRIBUTING.md) and [Development guide](https://pennylane.readthedocs.io/en/stable/development/guide.html) for more details.

## Support

*   **Source Code:** https://github.com/PennyLaneAI/pennylane
*   **Issue Tracker:** https://github.com/PennyLaneAI/pennylane/issues

For any issues, please report them on the GitHub issue tracker.

Join the [PennyLane Discussion Forum](https://discuss.pennylane.ai/) to connect with the quantum community.

## Authors

PennyLane is a collaborative effort by [many contributors](https://github.com/PennyLaneAI/pennylane/graphs/contributors).

Cite our paper if you use PennyLane in your research:

> Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical
> computations.* 2018. arXiv:1811.04968

## License

PennyLane is released under the Apache License, Version 2.0.