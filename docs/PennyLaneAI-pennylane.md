<h1 align="center">PennyLane: Your Gateway to Quantum Computing and Quantum Machine Learning</h1>

<p align="center">
  <a href="https://github.com/PennyLaneAI/pennylane">
    <img src="https://img.shields.io/github/stars/PennyLaneAI/pennylane?style=flat-square&logo=github" alt="GitHub stars">
  </a>
  <a href="https://pypi.org/project/PennyLane">
    <img src="https://img.shields.io/pypi/v/PennyLane.svg?style=flat-square&logo=pypi" alt="PyPI version">
  </a>
  <a href="https://pennylane.ai">
    <img src="https://img.shields.io/website?url=https%3A%2F%2Fpennylane.ai&style=flat-square&logo=pennylane" alt="PennyLane Website">
  </a>
</p>

<p align="center">
  <b>PennyLane is an open-source Python library that empowers researchers and developers to harness the power of quantum computing for quantum machine learning, quantum chemistry, and beyond.</b>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/pl-logo-lightmode.png#gh-light-mode-only" width="700px" alt="PennyLane Logo">
  <img src="./doc/_static/readme/pl-logo-darkmode.png#gh-dark-mode-only" width="700px" onerror="this.style.display='none'" alt="PennyLane Logo (Dark Mode)">
</p>

## Key Features

*   **Program Quantum Computers:** Build and simulate quantum circuits with a wide array of quantum gates and measurements.  Run on high-performance simulators or diverse quantum hardware devices with advanced features like mid-circuit measurements and error mitigation.
*   **Master Quantum Algorithms:** Explore and implement cutting-edge quantum algorithms from NISQ to fault-tolerant quantum computing. Analyze performance, visualize circuits, and leverage tools for quantum chemistry and algorithm development.
*   **Quantum Machine Learning Integration:** Seamlessly integrate with popular machine learning frameworks like PyTorch, TensorFlow, JAX, Keras, and NumPy. Define and train hybrid quantum-classical models using quantum-aware optimizers and hardware-compatible gradients.
*   **Quantum Datasets:** Access pre-simulated, high-quality quantum datasets to accelerate research and algorithm development. Browse existing datasets or contribute your own data.
*   **Compilation and Performance:** Benefit from experimental support for just-in-time (JIT) compilation with Catalyst, supporting adaptive circuits, real-time measurement feedback, and unbounded loops.

## Installation

PennyLane requires Python 3.11 or higher. Install PennyLane and its dependencies easily using pip:

```bash
python -m pip install pennylane
```

## Docker Support

Explore the [PennyLane Docker Hub page](https://hub.docker.com/u/pennylaneai) for Docker images and detailed information on PennyLane Docker support. For further details, see the [description here](https://docs.pennylane.ai/projects/lightning/en/stable/dev/docker.html).

## Getting Started

Kickstart your journey with PennyLane using our [quickstart guide](https://docs.pennylane.ai/en/stable/introduction/pennylane.html) designed to introduce key features and start building quantum circuits immediately.

Whether you're interested in quantum machine learning (QML), quantum computing, or quantum chemistry, PennyLane offers comprehensive resources:

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/research.png" align="right" width="350px">

### Key Resources:

*   [Research-oriented Demos](https://pennylane.ai/qml/demonstrations)
*   [Learn Quantum Programming](https://pennylane.ai/qml/) with the [Codebook](https://pennylane.ai/codebook/) and [Coding Challenges](https://pennylane.ai/challenges/)
*   [Frequently Asked Questions](https://pennylane.ai/faq)
*   [Glossary](https://pennylane.ai/qml/glossary)
*   [Videos](https://pennylane.ai/qml/videos)

You can also consult our [documentation](https://pennylane.readthedocs.io) for [quickstart
guides](https://pennylane.readthedocs.io/en/stable/introduction/pennylane.html) and comprehensive developer guides for creating [PennyLane-compatible quantum devices](https://pennylane.readthedocs.io/en/stable/development/plugins.html).

## Demos

Explore cutting-edge quantum algorithms and applications by exploring PennyLane's extensive collection of demos:

<a href="https://pennylane.ai/qml/demonstrations">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/demos.png" width="900px" alt="PennyLane Demos">
</a>

## Research Applications

PennyLane is driving innovation in quantum computing, quantum machine learning, and quantum chemistry. See how PennyLane is used in leading research:

*   **Quantum Computing**: [Fast quantum circuit cutting with randomized measurements](https://quantum-journal.org/papers/q-2023-03-02-934/)
*   **Quantum Machine Learning**: [Better than classical? The subtle art of benchmarking quantum machine learning models](https://arxiv.org/abs/2403.07059)
*   **Quantum Chemistry**: [Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization](https://quantum-journal.org/papers/q-2024-06-13-1371/)

We value your research!  Suggest features or report issues on [GitHub](https://github.com/PennyLaneAI/pennylane/issues/new?assignees=&labels=enhancement+%3Asparkles%3A&projects=&template=feature_request.yml) or visit our [website](https://pennylane.ai/research) for more information.

## Contributing

We welcome contributions.  Fork the PennyLane repository and submit a [pull request](https://help.github.com/articles/about-pull-requests/) with your improvements.  All contributors will be listed as authors. Those making significant contributions to the code will be listed on the PennyLane arXiv paper.

We encourage bug reports, feature suggestions, and links to interesting projects built using PennyLane.

See our [contributions page](https://github.com/PennyLaneAI/pennylane/blob/master/.github/CONTRIBUTING.md) and [Development guide](https://pennylane.readthedocs.io/en/stable/development/guide.html) for further details.

## Support

*   **Source Code:** [https://github.com/PennyLaneAI/pennylane](https://github.com/PennyLaneAI/pennylane)
*   **Issue Tracker:** [https://github.com/PennyLaneAI/pennylane/issues](https://github.com/PennyLaneAI/pennylane/issues)

For support, please post your issue on our GitHub issue tracker.

Join the [PennyLane Discussion Forum](https://discuss.pennylane.ai/) to connect with the quantum community, get support, and interact with our team.

Adhere to our [Code of Conduct](.github/CODE_OF_CONDUCT.md) for a welcoming environment.

## Authors

PennyLane is a collaborative effort by [many contributors](https://github.com/PennyLaneAI/pennylane/graphs/contributors).

If you are using PennyLane in your research, cite [our paper](https://arxiv.org/abs/1811.04968):

> Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical
> computations.* 2018. arXiv:1811.04968

## License

PennyLane is released under the Apache License, Version 2.0.
```
Key improvements and SEO considerations:

*   **Clear Title and Introduction:**  Uses a strong, SEO-friendly title and a one-sentence hook to grab attention.
*   **Concise and Focused Content:**  Keeps the content brief and to the point, using bullet points for key features and resources.
*   **Keyword Optimization:** Includes relevant keywords like "quantum computing," "quantum machine learning," "quantum chemistry," and "open source" throughout the README.
*   **Structured Headings:** Uses clear headings and subheadings to improve readability and SEO.
*   **Call to Actions:** Encourages users to contribute, report issues, and join the discussion forum.
*   **Visual Appeal:**  Includes links to badges for GitHub stars, PyPI version, and website, as well as well-placed images to enhance the user experience.
*   **Comprehensive Coverage:**  Covers all the important sections of the original README.
*   **Improved Formatting:**  Uses consistent Markdown formatting for better readability.
*   **Clear Installation Instructions:** Provides easy-to-follow installation steps.
*   **Strong Emphasis on Research:** Highlights the research applications of PennyLane and encourages researchers to contribute.
*   **SEO-Friendly Links:**  All links are descriptive.
*   **Concise Summaries:** Provides brief summaries of key features and resources.
*   **Mobile-Friendly Design:** By using proper formatting, the README is readable on mobile devices.