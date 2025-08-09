<h1 align="center">RD-Agent: Automating Machine Learning Engineering</h1>

<p align="center">
    <b>Revolutionize your R&D process with RD-Agent, the leading AI agent for automating machine learning engineering tasks!</b>
    <br>
    <a href="https://github.com/microsoft/RD-Agent">
        <img src="https://img.shields.io/github/stars/microsoft/RD-Agent?style=social" alt="GitHub stars">
    </a>
    <a href="https://discord.gg/ybQ97B6Jjy">
        <img src="https://img.shields.io/discord/1147121189911013515?label=Discord&logo=discord" alt="Discord">
    </a>
</p>

---

RD-Agent is an innovative multi-agent framework designed to automate and accelerate the full R&D lifecycle in machine learning engineering and quantitative finance.  It leverages the power of Large Language Models (LLMs) to automate critical aspects of the R&D process. Explore cutting-edge AI research and development with RD-Agent! ([Original Repo](https://github.com/microsoft/RD-Agent))

## Key Features

*   **Automated ML Engineering:** Automates the model and feature development process by iteratively proposing ideas, implementing them, and evaluating results.
*   **Data-Centric Focus:** Designed to optimize model performance by focusing on the data, supporting data-driven R&D.
*   **Multi-Agent Framework:** Leverages the power of multiple agents to automate complex tasks.
*   **Quantitative Finance Capabilities:** Includes a specialized agent (RD-Agent(Q)) for automating the full-stack research and development of quantitative strategies.
*   **Top Performance:** Currently the **top-performing machine learning engineering agent** on the MLE-bench benchmark.
*   **Modular Design:** Supports diverse scenarios, including general data science, Kaggle competitions, and financial applications.
*   **Easy to Use:** Simple setup and configuration with LiteLLM integration for flexible LLM selection.
*   **Comprehensive Documentation:** Extensive documentation, including demos, guides, and technical reports.

## Benchmarking and Performance

RD-Agent demonstrates superior performance in several machine learning engineering tasks:

*   **MLE-bench Leader:**  R&D-Agent outperforms other agents on the MLE-bench benchmark, showing a significant edge in various complexity levels.
    *   **R&D-Agent o1-preview:** Achieves impressive results across different complexity levels. (See the original README for detailed performance metrics).
    *   **R&D-Agent o3(R)+GPT-4.1(D):** Achieves impressive results across different complexity levels. (See the original README for detailed performance metrics).

## Getting Started

### Installation

1.  **Prerequisites:** Ensure you have Docker installed and configured correctly.  See the official [Docker installation instructions](https://docs.docker.com/engine/install/) and the README for verification steps.
2.  **Conda Environment:** Create and activate a conda environment:
    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```
3.  **Install RD-Agent:**
    ```bash
    pip install rdagent
    ```

### Configuration

1.  **API Keys:** Configure your preferred Chat and Embedding Models. We support LiteLLM for flexible model selection.  You can configure it with either a unified API base or separate bases for chat and embedding models. Examples are in the original README.
2.  **Health Check:** Verify your configuration:
    ```bash
    rdagent health_check
    ```

### Running Demos

Explore the following demos:

*   **Quantitative Finance:**
    *   Automated Quantitative Trading & Iterative Factors Model Joint Evolution
        ```bash
        rdagent fin_quant
        ```
    *   Automated Quantitative Trading & Iterative Factors Evolution
        ```bash
        rdagent fin_factor
        ```
    *   Automated Quantitative Trading & Iterative Model Evolution
        ```bash
        rdagent fin_model
        ```
    *   Automated Quantitative Trading & Factors Extraction from Financial Reports
        ```bash
        rdagent fin_factor_report --report_folder=<Your financial reports folder path>
        ```
        (See original README for preparing financial reports)
*   **General Model Research & Development Copilot:**
    ```bash
    rdagent general_model <Your paper URL>
    ```
    (See original README for examples)
*   **Data Science Agent (Kaggle Example):**
    ```bash
    rdagent data_science --competition tabular-playground-series-dec-2021
    ```
    (See original README for data science environment setup)
*   **Automated Medical Prediction Model Evolution**:
    ```bash
    rdagent data_science --competition arf-12-hours-prediction-task
    ```

### Monitoring Results

Monitor the application results:
```bash
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

## Framework Overview

RD-Agent's framework focuses on automating the core R&D process through:

*   **Idea Proposal:** Generation of new hypotheses.
*   **Development:** Implementation and execution of ideas.
*   **Benchmarking:** Evaluating R&D abilities.

For deeper insights, refer to the [Framework section](#-framework) in the original README.

## Resources

*   [🖥️ Live Demo](https://rdagent.azurewebsites.net/)
*   [🎥 Demo Video](https://rdagent.azurewebsites.net/factor_loop) | [▶️YouTube](https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR)
*   [📖 Documentation](https://rdagent.readthedocs.io/en/latest/index.html)
*   [📄 Tech Report](https://aka.ms/RD-Agent-Tech-Report)
*   [📃 Papers](#-paperwork-list)
*   [⚡ Quick Start](#-quick-start)

## Contribute

We welcome contributions! Please see the [Contributing Guide](CONTRIBUTING.md) in the original repository for instructions.

<img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>

<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

## Legal Disclaimer

Refer to the [Legal Disclaimer](#-legal-disclaimer) in the original README.

---

**Discover the future of machine learning engineering with RD-Agent!**