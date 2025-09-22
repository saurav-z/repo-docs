<!-- PennyLane Banner - Keeping this at the top for SEO -->
<p align="center">
<img width="200" height="46" alt="PennyLane Logo" src="https://github.com/user-attachments/assets/643e69cd-3d41-42f9-a9e8-ac113e8b6de3" />
</p>

<!-- Update Announcement - Keeping this prominent as requested -->
<h2 align="center"> The 2025 Quantum Open Source Software Survey is now open. 👉 <a href="https://unitary.foundation/posts/2025_qoss_survey/" target="_blank">Take the survey!</a> 👈</h2>

<p align="center">
Take 10 minutes to share your voice and help build a better quantum computing ecosystem!
</p>

<p align="center"><img width="200" alt="UnitaryFundSurvey" src="https://assets.cloud.pennylane.ai/pennylane_website/spotlights/Spotlight_UnitaryFundSurvey.png" /></p>

<p align="center">
Survey closes October 3, 2025
</p>

<br>
<br>
<br>

<!-- SEO Optimization: Shields and Links  -->
<p align="center">
  <!-- Tests (GitHub actions) -->
  <a href="https://github.com/PennyLaneAI/pennylane/actions?query=workflow%3ATests">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/PennyLane/tests.yml?branch=master&style=flat-square" alt="Build Status"/>
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

<!-- Main Body - Concise & Optimized -->
<!-- Title and Hook -->
<h1 align="center">PennyLane: The Open-Source Quantum Computing Library</h1>
<p align="center"><b>PennyLane is a versatile Python library empowering researchers to explore quantum computing, machine learning, and chemistry.</b></p>

<!-- Main Logo -  Keep prominent -->
<p align="center">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/pl-logo-lightmode.png#gh-light-mode-only" width="700px" alt="PennyLane Logo Light Mode">
    <!-- Dark mode image using a relative import.  Handles PyPI gracefully. -->
    <img src="./doc/_static/readme/pl-logo-darkmode.png#gh-dark-mode-only" width="700px" onerror="this.style.display='none'" alt="PennyLane Logo Dark Mode"/>
</p>

<!-- Key Features - Bulleted for readability and SEO -->
## Key Features

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/code.png" width="400px" align="right" alt="Code Snippet">

*   <b>Quantum Circuit Programming:</b> Construct quantum circuits with diverse gates, state preparations, and measurements. Run on high-performance simulators and various hardware devices with advanced features like mid-circuit measurements and error mitigation.
*   <b>Quantum Algorithm Development:</b> Explore algorithms for NISQ and fault-tolerant quantum computing. Analyze performance, visualize circuits, and access tools for quantum chemistry and algorithm development.
*   <b>Quantum Machine Learning Integration:</b> Integrate with PyTorch, TensorFlow, JAX, Keras, or NumPy to build and train hybrid models. Utilize quantum-aware optimizers and hardware-compatible gradients for advanced research.
*   <b>Quantum Datasets:</b> Access pre-simulated datasets to expedite research and algorithm development. Browse datasets or contribute your own.
*   <b>Compilation and Performance:</b> Experimental support for just-in-time compilation with advanced features like adaptive circuits, real-time measurement feedback, and unbounded loops.

  <a href="https://pennylane.ai/features/">Explore more features on the PennyLane website.</a>

## Installation

Install PennyLane with Python 3.11+ and pip:

```bash
python -m pip install pennylane
```

## Docker Support

Find Docker images on the [PennyLane Docker Hub page](https://hub.docker.com/u/pennylaneai).  Detailed information about PennyLane Docker support is [here](https://docs.pennylane.ai/projects/lightning/en/stable/dev/docker.html).

## Getting Started

Begin your quantum journey with the [quickstart guide](https://docs.pennylane.ai/en/stable/introduction/pennylane.html) and explore quantum machine learning (QML), quantum computing, and quantum chemistry with PennyLane's comprehensive tools and resources.

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/research.png" align="right" width="350px" alt="Research Icon">

### Key Resources:

*   [Research-oriented Demos](https://pennylane.ai/qml/demonstrations)
*   [Learn Quantum Programming](https://pennylane.ai/qml/) with the [Codebook](https://pennylane.ai/codebook/) and [Coding Challenges](https://pennylane.ai/challenges/)
*   [Frequently Asked Questions](https://pennylane.ai/faq)
*   [Glossary](https://pennylane.ai/qml/glossary)
*   [Videos](https://pennylane.ai/qml/videos)

Consult the [documentation](https://pennylane.readthedocs.io) for quickstart guides and detailed developer guides for creating PennyLane-compatible quantum devices.

## Demos

Explore cutting-edge quantum algorithms with PennyLane and quantum hardware through demos. [Explore PennyLane demos](https://pennylane.ai/qml/demonstrations).

<a href="https://pennylane.ai/qml/demonstrations">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/demos.png" width="900px" alt="PennyLane Demos">
</a>

Submit your demo via the [demo submission guide](https://pennylane.ai/qml/demos_submission).

## Research Applications

PennyLane is at the forefront of quantum computing research. Explore how PennyLane is used for research in publications:

*   **Quantum Computing**: [Fast quantum circuit cutting with randomized measurements](https://quantum-journal.org/papers/q-2023-03-02-934/)
*   **Quantum Machine Learning**: [Better than classical? The subtle art of benchmarking quantum machine learning models](https://arxiv.org/abs/2403.07059)
*   **Quantum Chemistry**: [Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization](https://quantum-journal.org/papers/q-2024-06-13-1371/)

We welcome your feedback: [GitHub](https://github.com/PennyLaneAI/pennylane/issues/new?assignees=&labels=enhancement+%3Asparkles%3A&projects=&template=feature_request.yml) or [website](https://pennylane.ai/research).

## Contributing to PennyLane

Contribute to PennyLane by forking the repository and submitting a [pull request](https://help.github.com/articles/about-pull-requests/).  All contributors are recognized.

Report bugs, suggest features, and share projects.  See the [contributions page](https://github.com/PennyLaneAI/pennylane/blob/master/.github/CONTRIBUTING.md) and [Development guide](https://pennylane.readthedocs.io/en/stable/development/guide.html).

## Support

*   **Source Code:** [https://github.com/PennyLaneAI/pennylane](https://github.com/PennyLaneAI/pennylane)
*   **Issue Tracker:** [https://github.com/PennyLaneAI/pennylane/issues](https://github.com/PennyLaneAI/pennylane/issues)

Report issues on the GitHub issue tracker.

Join the [PennyLane Discussion Forum](https://discuss.pennylane.ai/) for support and collaboration.

Read the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## Authors

PennyLane is created by [many contributors](https://github.com/PennyLaneAI/pennylane/graphs/contributors).

If you use PennyLane in your research, please cite [our paper](https://arxiv.org/abs/1811.04968):

> Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968

## License

PennyLane is **open source** and licensed under the Apache License, Version 2.0.
```

Key improvements and SEO considerations:

*   **Clear Title and Hook:** Added a strong title and a concise hook to immediately grab attention.
*   **SEO-Friendly Structure:** Used headings (H1, H2) and bullet points for better readability and search engine indexing.
*   **Keyword Optimization:** Naturally incorporated relevant keywords like "quantum computing," "quantum machine learning," "quantum chemistry," "Python library," and "open source" throughout the text.
*   **Concise Descriptions:**  Simplified and clarified descriptions for each key feature.
*   **Alt Text for Images:**  Added `alt` text to all images to improve accessibility and SEO.
*   **Internal Linking:**  Linked to key resources within the PennyLane ecosystem.
*   **Call to Action:** Used strong calls to action like "Explore more features," "Get started," and "Join the discussion."
*   **Clean Code Blocks:** Ensured code snippets are properly formatted.
*   **Removed Redundancy:** Consolidated some sections to avoid repetition.
*   **Maintained Original Content:** Preserved the core information while optimizing for clarity and searchability.
*   **Markdown Compliance:** Made sure the formatting is valid markdown.
*   **Added link to the repo:** Added the link back to the original repo.
*   **Removed redundant images.**
*   **Clear license information.**