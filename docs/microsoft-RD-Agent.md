<h1 align="center">RD-Agent: Automate Your Data-Driven R&D with AI</h1>

<div align="center">
  <a href="https://github.com/microsoft/RD-Agent">
    <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:50%;">
  </a>
</div>

**RD-Agent empowers researchers and engineers to automate the most critical aspects of the R&D process, leading to groundbreaking discoveries and improved productivity.**

[🖥️ Live Demo](https://rdagent.azurewebsites.net/) | [🎥 Demo Video](https://rdagent.azurewebsites.net/factor_loop) | [📖 Documentation](https://rdagent.readthedocs.io/en/latest/index.html) | [📄 Tech Report](https://aka.ms/RD-Agent-Tech-Report) | [▶️ YouTube Channel](https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR)

---

## Key Features

*   **Automated R&D:**  R&D-Agent automates key steps in the R&D process, including idea generation, implementation, and evaluation.
*   **Multi-Agent Framework:**  RD-Agent is built on a multi-agent architecture, allowing for collaboration and specialization in various tasks.
*   **Data-Centric Approach:** The framework focuses on data-driven scenarios, enabling more effective model and data development.
*   **Extensible and Flexible:** Designed to easily integrate with various datasets, models, and tools, making it adaptable to diverse R&D challenges.
*   **Top-Performing on MLE-bench:** R&D-Agent achieves state-of-the-art results on the MLE-bench benchmark, demonstrating its ability to excel in real-world ML engineering tasks.

## Quick Start

*   RD-Agent currently supports Linux.
*   **Installation:**
    *   Install Docker.  See the [official Docker page](https://docs.docker.com/engine/install/) for installation instructions.
    *   Create and activate a Conda environment.
    *   Install the RD-Agent package using `pip install rdagent`.
*   **Configuration:**  Set up your environment variables, including your LLM provider details (e.g., OpenAI API key, Azure OpenAI API key).  We now fully support **[LiteLLM](https://github.com/BerriAI/litellm)** as our default backend for integration with multiple LLM providers.
*   **Run the Application:** Execute `rdagent health_check` to verify your configuration. Then, run one of the provided demo applications (e.g., `rdagent fin_quant`) to see RD-Agent in action. See the [documentation](https://rdagent.readthedocs.io/en/latest/index.html) for more details.

### Available Demos

*   Automated Quantitative Trading & Iterative Factors Model Joint Evolution (Qlib)
*   Automated Quantitative Trading & Iterative Factors Evolution (Qlib)
*   Automated Quantitative Trading & Iterative Model Evolution (Qlib)
*   Automated Quantitative Trading & Factors Extraction from Financial Reports
*   Automated Model Research & Development Copilot (Paper Implementation)
*   Automated Medical Prediction Model Evolution
*   Automated Kaggle Model Tuning & Feature Engineering

### Monitoring Results

Use the `rdagent ui` command to monitor the application results via the UI.  Ensure the appropriate port is not occupied before running.

##  Scenarios and Applications

RD-Agent is designed to automate key stages in data-driven R&D across various domains:

*   **Finance:** Automated Quant Factory, Iterative Factor and Model Optimization
*   **Medical:** Automated Model Evolution for Predictions
*   **General Data Science:**  Automated Paper Reading & Implementation, Kaggle Competition Agent
*   Explore the [Live Demo](https://rdagent.azurewebsites.net/)

## Framework Overview

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

The RD-Agent framework is built on key components:

*   **Benchmark:**  Evaluating the abilities of R&D agents.
*   **Idea Proposal:** Generating new ideas or refining existing ones.
*   **Development:** Implementing and executing ideas.

For detailed research questions, please consult the [documentation](https://rdagent.readthedocs.io/).

## Papers and Publications

*   [Overall Technical Report](https://arxiv.org/abs/2505.14738)
*   [Benchmark](https://arxiv.org/abs/2404.11276)
*   [Research](https://rdagent.azurewebsites.net)
*   [Development](https://arxiv.org/abs/2407.18690)
*   [Deep Application in Diverse Scenarios](https://arxiv.org/abs/2505.15155)

## Contributing

We welcome contributions!  Please refer to the [Contributing Guide](CONTRIBUTING.md).

<img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>

<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

## Legal Disclaimer

(Include your legal disclaimer here, as provided in the original README)

---

**[Visit the RD-Agent GitHub Repository](https://github.com/microsoft/RD-Agent)** to explore the code, contribute, and learn more.