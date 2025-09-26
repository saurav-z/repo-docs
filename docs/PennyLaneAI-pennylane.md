<p align="center">
<img width="200" height="46" alt="PennyLane Logo" src="https://github.com/user-attachments/assets/643e69cd-3d41-42f9-a9e8-ac113e8b6de3" />
</p>

<h1 align="center">PennyLane: Your Gateway to Quantum Computing, Quantum Machine Learning, and Quantum Chemistry</h1>

<p align="center">
  <b>Explore the power of quantum computing with PennyLane, the open-source Python library for quantum programming.</b>
</p>

<p align="center">
  <a href="https://github.com/PennyLaneAI/pennylane">
    <img src="https://img.shields.io/github/stars/PennyLaneAI/pennylane?style=flat-square&logo=github" alt="GitHub stars">
  </a>
  <a href="https://pypi.org/project/PennyLane">
    <img src="https://img.shields.io/pypi/v/PennyLane.svg?style=flat-square&logo=pypi" alt="PyPI version">
  </a>
  <a href="https://github.com/PennyLaneAI/pennylane/actions?query=workflow%3ATests">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/PennyLane/tests.yml?branch=master&style=flat-square&label=tests&logo=github" alt="Tests status">
  </a>
  <a href="https://codecov.io/gh/PennyLaneAI/pennylane">
    <img src="https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane/master.svg?logo=codecov&style=flat-square" alt="Codecov coverage">
  </a>
  <a href="https://docs.pennylane.ai/en/latest">
    <img src="https://readthedocs.com/projects/xanaduai-pennylane/badge/?version=latest&style=flat-square" alt="Read the Docs">
  </a>
  <a href="https://discuss.pennylane.ai">
    <img src="https://img.shields.io/discourse/https/discuss.pennylane.ai/posts.svg?logo=discourse&style=flat-square" alt="Discourse Forum">
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/pypi/l/PennyLane.svg?logo=apache&style=flat-square" alt="License">
  </a>

</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/pl-logo-lightmode.png#gh-light-mode-only" width="700px" alt="PennyLane Logo Light Mode">
  <img src="./doc/_static/readme/pl-logo-darkmode.png#gh-dark-mode-only" width="700px" onerror="this.style.display='none'" alt="PennyLane Logo Dark Mode"/>
</p>

## Key Features

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/code.png" width="400px" align="right" alt="Code Snippet">

PennyLane empowers researchers and developers to explore the forefront of quantum technologies:

*   **Program Quantum Computers:** Build and simulate quantum circuits with a vast array of gates, measurements, and state preparations. Run your circuits on high-performance simulators or connect to a growing ecosystem of quantum hardware devices. Includes advanced features like mid-circuit measurements and error mitigation.

*   **Master Quantum Algorithms:** Develop, analyze, and visualize quantum algorithms. PennyLane supports a wide range of quantum algorithms from NISQ to fault-tolerant quantum computing. Access tools and examples for quantum chemistry and algorithm development.

*   **Quantum Machine Learning:** Seamlessly integrate with popular machine learning frameworks like PyTorch, TensorFlow, JAX, Keras, and NumPy. Define and train hybrid quantum-classical models using quantum-aware optimizers and hardware-compatible gradients.

*   **Quantum Datasets:** Accelerate your research with pre-simulated, high-quality quantum datasets. Easily access and utilize existing datasets or contribute your own.

*   **Compilation and Performance:** Utilize experimental support for just-in-time compilation to optimize your workflows. Benefit from features like adaptive circuits, real-time measurement feedback, and unbounded loops. Explore [Catalyst](https://github.com/pennylaneai/catalyst) for more details.

For a complete list of features and capabilities, visit the [PennyLane website](https://pennylane.ai/features/).

## Installation

PennyLane requires Python 3.11 or later. Install PennyLane and its dependencies using `pip`:

```bash
python -m pip install pennylane
```

## Docker Support

PennyLane provides Docker images for easy deployment and use. Find the images and documentation on the [PennyLane Docker Hub page](https://hub.docker.com/u/pennylaneai).  For more details on Docker support, see the [description here](https://docs.pennylane.ai/projects/lightning/en/stable/dev/docker.html).

## Getting Started

Jumpstart your quantum journey with the [PennyLane quickstart guide](https://docs.pennylane.ai/en/stable/introduction/pennylane.html). Learn key concepts and start building quantum circuits immediately.

### Key Resources for Exploration:

*   [Research-oriented Demos](https://pennylane.ai/qml/demonstrations)
*   [Learn Quantum Programming](https://pennylane.ai/qml/) with the [Codebook](https://pennylane.ai/codebook/) and [Coding Challenges](https://pennylane.ai/challenges/)
*   [Frequently Asked Questions](https://pennylane.ai/faq)
*   [Glossary](https://pennylane.ai/qml/glossary)
*   [Videos](https://pennylane.ai/qml/videos)

For detailed information, consult the [PennyLane documentation](https://pennylane.readthedocs.io). Explore [quickstart guides](https://pennylane.readthedocs.io/en/stable/introduction/pennylane.html) and developer resources for creating your own PennyLane-compatible quantum devices.

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/research.png" align="right" width="350px" alt="Research Icon">

## Demos

Discover real-world applications and dive into quantum computing through a collection of demonstrations.  Explore the [PennyLane demos](https://pennylane.ai/qml/demonstrations).

<a href="https://pennylane.ai/qml/demonstrations">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/demos.png" width="900px" alt="Demos">
</a>

Interested in contributing your own demo?  See the [demo submission guide](https://pennylane.ai/qml/demos_submission).

## Research Applications

PennyLane is instrumental in cutting-edge quantum research across various domains. Explore recent publications:

*   **Quantum Computing:** [Fast quantum circuit cutting with randomized measurements](https://quantum-journal.org/papers/q-2023-03-02-934/)
*   **Quantum Machine Learning:** [Better than classical? The subtle art of benchmarking quantum machine learning models](https://arxiv.org/abs/2403.07059)
*   **Quantum Chemistry:** [Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization](https://quantum-journal.org/papers/q-2024-06-13-1371/)

Your research drives PennyLane's development.  Share your feature requests on [GitHub](https://github.com/PennyLaneAI/pennylane/issues/new?assignees=&labels=enhancement+%3Asparkles%3A&projects=&template=feature_request.yml) or on our [website](https://pennylane.ai/research).

## Contributing

We welcome contributions!  Start by forking the PennyLane repository and submitting a [pull request](https://help.github.com/articles/about-pull-requests/). Your contributions will be recognized.

Report bugs, suggest enhancements, or share cool projects built with PennyLane.

Learn more on our [contributions page](https://github.com/PennyLaneAI/pennylane/blob/master/.github/CONTRIBUTING.md) and in the [Development guide](https://pennylane.readthedocs.io/en/stable/development/guide.html).

## Support

*   **Source Code:** [https://github.com/PennyLaneAI/pennylane](https://github.com/PennyLaneAI/pennylane)
*   **Issue Tracker:** [https://github.com/PennyLaneAI/pennylane/issues](https://github.com/PennyLaneAI/pennylane/issues)

Report any issues using the GitHub issue tracker.

Join the [PennyLane Discussion Forum](https://discuss.pennylane.ai/) for support, community interaction, and direct engagement with the PennyLane team.

We are committed to a welcoming and safe environment for all.  Please review the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## Authors

PennyLane is a collaborative effort.  See the [contributors](https://github.com/PennyLaneAI/pennylane/graphs/contributors).

Cite our paper if you use PennyLane in your research:

> Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical
> computations.* 2018. arXiv:1811.04968

## License

PennyLane is **free** and **open source**, released under the Apache License, Version 2.0.