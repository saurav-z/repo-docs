# PennyLane: The Open-Source Quantum Computing Library

**Unlock the power of quantum computing, machine learning, and chemistry with PennyLane, the versatile Python library that bridges the gap between research and real-world applications.**  [Explore the PennyLane Repository](https://github.com/PennyLaneAI/pennylane)

<p align="center">
  <!-- Tests (GitHub actions) -->
  <a href="https://github.com/PennyLaneAI/pennylane/actions?query=workflow%3ATests">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/PennyLane/tests.yml?branch=master&style=flat-square" />
  </a>
  <!-- CodeCov -->
  <a href="https://codecov.io/gh/PennyLaneAI/pennylane">
    <img src="https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane/master.svg?logo=codecov&style=flat-square" />
  </a>
  <!-- ReadTheDocs -->
  <a href="https://docs.pennylane.ai/en/latest">
    <img src="https://readthedocs.com/projects/xanaduai-pennylane/badge/?version=latest&style=flat-square" />
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/PennyLane">
    <img src="https://img.shields.io/pypi/v/PennyLane.svg?style=flat-square" />
  </a>
  <!-- Forum -->
  <a href="https://discuss.pennylane.ai">
    <img src="https://img.shields.io/discourse/https/discuss.pennylane.ai/posts.svg?logo=discourse&style=flat-square" />
  </a>
  <!-- License -->
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/pypi/l/PennyLane.svg?logo=apache&style=flat-square" />
  </a>
</p>

<p align="center">
  <a href="https://pennylane.ai">PennyLane</a> is a cross-platform Python library for
  <a href="https://pennylane.ai/qml/quantum-computing/">quantum computing</a>,
  <a href="https://pennylane.ai/qml/quantum-machine-learning/">quantum machine learning</a>,
  and
  <a href="https://pennylane.ai/qml/quantum-chemistry/">quantum chemistry</a>.
</p>

<p align="center">
  The definitive open-source framework for quantum programming. Built by researchers, for research.
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/pl-logo-lightmode.png#gh-light-mode-only" width="700px">
    <!--
    Use a relative import for the dark mode image. When loading on PyPI, this
    will fail automatically and show nothing.
    -->
    <img src="./doc/_static/readme/pl-logo-darkmode.png#gh-dark-mode-only" width="700px" onerror="this.style.display='none'" alt=""/>
</p>


## Key Features of PennyLane

PennyLane offers a comprehensive suite of tools for quantum research and development:

*   **Program Quantum Computers:**  Build and execute quantum circuits using a wide array of gates, measurements, and state preparations.  Benefit from support for high-performance simulators and various hardware devices, including advanced features like mid-circuit measurements and error mitigation.

*   **Master Quantum Algorithms:**  Explore and implement cutting-edge algorithms, from NISQ to fault-tolerant models. Analyze performance, visualize circuits, and utilize tools for quantum chemistry and algorithm development.

*   **Quantum Machine Learning Integration:**  Seamlessly integrate with PyTorch, TensorFlow, JAX, Keras, and NumPy to define and train hybrid quantum-classical models. Leverage quantum-aware optimizers and hardware-compatible gradients for advanced research tasks.

*   **Quantum Datasets:** Access pre-simulated quantum datasets to accelerate research and algorithm development.  Explore the available datasets and contribute your own.

*   **Compilation and Performance:**  Experimental support for just-in-time compilation with Catalyst. Compile your entire hybrid workflow with support for advanced features such as adaptive circuits, real-time measurement feedback, and unbounded loops.

For more detailed information, visit the [PennyLane website](https://pennylane.ai/features/).

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/code.png" width="400px" align="right">

## Installation

PennyLane requires Python 3.11 or later. Install it and its dependencies using pip:

```bash
python -m pip install pennylane
```

## Docker Support

PennyLane provides Docker images, available on the [PennyLane Docker Hub page](https://hub.docker.com/u/pennylaneai).  Refer to the [documentation](https://docs.pennylane.ai/projects/lightning/en/stable/dev/docker.html) for detailed information on using Docker with PennyLane.

## Getting Started

Start your quantum journey with the [PennyLane Quickstart Guide](https://docs.pennylane.ai/en/stable/introduction/pennylane.html), designed to get you building quantum circuits quickly.

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/research.png" align="right" width="350px">

## Key Resources

*   [Research-oriented Demos](https://pennylane.ai/qml/demonstrations)
*   [Learn Quantum Programming](https://pennylane.ai/qml/) with the [Codebook](https://pennylane.ai/codebook/) and [Coding Challenges](https://pennylane.ai/challenges/)
*   [Frequently Asked Questions](https://pennylane.ai/faq)
*   [Glossary](https://pennylane.ai/qml/glossary)
*   [Videos](https://pennylane.ai/qml/videos)
*   [Documentation](https://pennylane.readthedocs.io)

## Demos

Explore cutting-edge quantum algorithms with PennyLane and quantum hardware through interactive demos.

[Explore PennyLane demos](https://pennylane.ai/qml/demonstrations).

<a href="https://pennylane.ai/qml/demonstrations">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/demos.png" width="900px">
</a>

Contribute your own demo by following the [demo submission guide](https://pennylane.ai/qml/demos_submission).

## Research Applications

PennyLane is at the forefront of research in quantum computing, quantum machine learning, and quantum chemistry. Explore how PennyLane is used for research in the following publications:

- **Quantum Computing**: [Fast quantum circuit cutting with randomized measurements](https://quantum-journal.org/papers/q-2023-03-02-934/)

- **Quantum Machine Learning**: [Better than classical? The subtle art of benchmarking quantum machine learning models](https://arxiv.org/abs/2403.07059)

- **Quantum Chemistry**: [Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization](https://quantum-journal.org/papers/q-2024-06-13-1371/)

Share what features you need for your research on [GitHub](https://github.com/PennyLaneAI/pennylane/issues/new?assignees=&labels=enhancement+%3Asparkles%3A&projects=&template=feature_request.yml) or on our [website](https://pennylane.ai/research).

## Contributing

Contribute to PennyLane by forking the repository and submitting a [pull request](https://help.github.com/articles/about-pull-requests/). See the [contributions page](https://github.com/PennyLaneAI/pennylane/blob/master/.github/CONTRIBUTING.md) and [Development guide](https://pennylane.readthedocs.io/en/stable/development/guide.html) for details.

## Support

*   **Source Code:** [https://github.com/PennyLaneAI/pennylane](https://github.com/PennyLaneAI/pennylane)
*   **Issue Tracker:** [https://github.com/PennyLaneAI/pennylane/issues](https://github.com/PennyLaneAI/pennylane/issues)

Report issues on the GitHub issue tracker.

Join the [PennyLane Discussion Forum](https://discuss.pennylane.ai/) for community support, engagement, and direct interaction with the PennyLane team.

Please adhere to the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## Authors

PennyLane is developed by [many contributors](https://github.com/PennyLaneAI/pennylane/graphs/contributors).

Cite our paper if you use PennyLane in your research:

> Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical
> computations.* 2018. arXiv:1811.04968

## License

PennyLane is licensed under the Apache License, Version 2.0.