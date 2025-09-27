<!--  Remove the old survey information to keep the README focused and current -->

<p align="center">
<img width="200" height="46" alt="PennyLane Logo" src="https://github.com/user-attachments/assets/643e69cd-3d41-42f9-a9e8-ac113e8b6de3" />
</p>

<h1 align="center">PennyLane: The Open-Source Quantum Computing Library</h1>

<p align="center">
Build and explore the future of quantum computing with PennyLane, the definitive open-source framework for quantum programming, quantum machine learning, and quantum chemistry.  <a href="https://github.com/PennyLaneAI/pennylane">Explore the code on GitHub!</a>
</p>

<!-- Shields -->
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
    <img src="https://img.shields.io/discourse/https/discuss.pennylane.ai/posts.svg?logo=discourse&style=flat-square" alt="Discourse Forum"/>
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

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/code.png" width="400px" align="right" alt="Code Snippet">

PennyLane empowers researchers and developers to explore quantum computing, quantum machine learning, and quantum chemistry with a powerful and versatile toolkit.  Here are some of the key features:

*   **Program Quantum Computers:** Build quantum circuits using a wide range of gates, state preparations, and measurements. Run your circuits on high-performance simulators or various hardware devices ([plugins](https://pennylane.ai/plugins/)). Includes advanced features like mid-circuit measurements and error mitigation.
*   **Master Quantum Algorithms:** Develop and analyze algorithms from NISQ to fault-tolerant quantum computing. Visualize circuits, access tools for [quantum chemistry](https://docs.pennylane.ai/en/stable/introduction/chemistry.html) and [algorithm development](https://pennylane.ai/search/?contentType=DEMO&categories=algorithms&sort=publication_date).
*   **Quantum Machine Learning:** Integrate with PyTorch, TensorFlow, JAX, Keras, or NumPy to define and train hybrid models. Leverage quantum-aware optimizers and hardware-compatible gradients for advanced research tasks.  Get started with the [quantum machine learning quickstart](https://docs.pennylane.ai/en/stable/introduction/interfaces.html).
*   **Quantum Datasets:** Access high-quality, pre-simulated datasets to accelerate algorithm development and reduce time-to-research.  [Browse the datasets](https://pennylane.ai/datasets/) or contribute your own.
*   **Compilation and Performance:** Experimental support for just-in-time compilation. Compile your entire hybrid workflow, with support for advanced features such as adaptive circuits, real-time measurement feedback, and unbounded loops. See [Catalyst](https://github.com/pennylaneai/catalyst) for more details.

Explore more features on the [PennyLane website](https://pennylane.ai/features/).

## Installation

PennyLane requires Python version 3.11 or higher. Install PennyLane and its dependencies using pip:

```bash
python -m pip install pennylane
```

## Docker Support

Find PennyLane Docker images on the [PennyLane Docker Hub page](https://hub.docker.com/u/pennylaneai), and review the [PennyLane Docker support documentation](https://docs.pennylane.ai/projects/lightning/en/stable/dev/docker.html).

## Getting Started

Quickly get up and running with PennyLane by following our [quickstart guide](https://docs.pennylane.ai/en/stable/introduction/pennylane.html).

### Key Resources:

*   [Research-oriented Demos](https://pennylane.ai/qml/demonstrations)
*   [Learn Quantum Programming](https://pennylane.ai/qml/) with the [Codebook](https://pennylane.ai/codebook/) and [Coding Challenges](https://pennylane.ai/challenges/)
*   [Frequently Asked Questions](https://pennylane.ai/faq)
*   [Glossary](https://pennylane.ai/qml/glossary)
*   [Videos](https://pennylane.ai/qml/videos)

Also, check out our [documentation](https://pennylane.readthedocs.io) and developer guides on [how to write your own](https://pennylane.readthedocs.io/en/stable/development/plugins.html) PennyLane-compatible quantum device.

## Demos

Explore cutting-edge algorithms and applications through a collection of interactive demos.

[Explore PennyLane demos](https://pennylane.ai/qml/demonstrations).

<a href="https://pennylane.ai/qml/demonstrations">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/demos.png" width="900px" alt="Demos">
</a>

If you want to contribute your own demo, see our [demo submission guide](https://pennylane.ai/qml/demos_submission).

## Research Applications

PennyLane is at the forefront of research in quantum computing, quantum machine learning, and quantum chemistry. Explore how PennyLane is used for research in the following publications:

*   **Quantum Computing**: [Fast quantum circuit cutting with randomized measurements](https://quantum-journal.org/papers/q-2023-03-02-934/)
*   **Quantum Machine Learning**: [Better than classical? The subtle art of benchmarking quantum machine learning models](https://arxiv.org/abs/2403.07059)
*   **Quantum Chemistry**: [Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization](https://quantum-journal.org/papers/q-2024-06-13-1371/)

Share your research feature requests on [GitHub](https://github.com/PennyLaneAI/pennylane/issues/new?assignees=&labels=enhancement+%3Asparkles%3A&projects=&template=feature_request.yml) or our [website](https://pennylane.ai/research).

## Contributing to PennyLane

We welcome contributions!  Fork the repository and submit a [pull request](https://help.github.com/articles/about-pull-requests/) with your changes.

See our [contributions page](https://github.com/PennyLaneAI/pennylane/blob/master/.github/CONTRIBUTING.md) and our [Development guide](https://pennylane.readthedocs.io/en/stable/development/guide.html) for details.

## Support

*   **Source Code:** https://github.com/PennyLaneAI/pennylane
*   **Issue Tracker:** https://github.com/PennyLaneAI/pennylane/issues

Report issues on our GitHub issue tracker.

Join the [PennyLane Discussion Forum](https://discuss.pennylane.ai/) to connect with the quantum community, get support, and collaborate.

Please review and respect the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## Authors

PennyLane is developed by [many contributors](https://github.com/PennyLaneAI/pennylane/graphs/contributors).

If you use PennyLane in your research, please cite [our paper](https://arxiv.org/abs/1811.04968):

> Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical
> computations.* 2018. arXiv:1811.04968

## License

PennyLane is **free** and **open source**, released under the Apache License, Version 2.0.
```
Key improvements and explanations:

*   **SEO Optimization:**  Added keywords like "quantum computing," "quantum machine learning," "quantum chemistry," and "open source" in the title, intro, and key feature descriptions.  Included links to relevant external resources.  Used headings to structure the information logically.
*   **Hook:** Created a compelling one-sentence introduction that clearly states what PennyLane is and why it's important.
*   **Readability:**  Improved formatting for better readability.  Used bullet points for key features, clearer descriptions, and concise language.
*   **Summarization:** Condensed the original content while retaining all essential information. Removed the survey information because it's no longer relevant.
*   **Structure:**  Organized the README with clear headings and subheadings to improve navigation and understanding.
*   **Links:**  Ensured all relevant links are present and correctly formatted.  Added alt text to images for accessibility and SEO.
*   **Conciseness:** Removed redundant information.
*   **Call to Action:** Encouraged users to explore the code on GitHub.
*   **Up-to-date:**  The README is focused on current aspects of PennyLane.
*   **Accessibility:** Added `alt` tags to all images to improve accessibility.

This improved README is more informative, easier to understand, and optimized for both users and search engines.  It presents a clear and concise overview of PennyLane and its capabilities.