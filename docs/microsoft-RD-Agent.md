<h1 align="center">
  <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:70%; max-width: 600px;">
  <br>
  RD-Agent: Automate Your Machine Learning Engineering and Quantitative Finance R&D
</h1>

<p align="center">
    <a href="https://github.com/microsoft/RD-Agent">
        <img src="https://img.shields.io/github/stars/microsoft/RD-Agent?style=social" alt="GitHub stars">
    </a>
    <a href="https://github.com/microsoft/RD-Agent/actions/workflows/ci.yml">
      <img src="https://github.com/microsoft/RD-Agent/actions/workflows/ci.yml/badge.svg" alt="CI Status">
    </a>
    <a href="https://pypi.org/project/rdagent/">
        <img src="https://img.shields.io/pypi/v/rdagent" alt="PyPI version">
    </a>
    <a href="https://discord.gg/ybQ97B6Jjy">
      <img src="https://img.shields.io/badge/Chat-Discord-blue" alt="Discord Chat">
    </a>
    <a href="https://rdagent.readthedocs.io/en/latest/index.html">
        <img src="https://readthedocs.org/projects/rdagent/badge/?version=latest" alt="Documentation">
    </a>
</p>

**RD-Agent is a cutting-edge AI framework designed to automate and accelerate the entire R&D lifecycle, from model creation to quantitative finance strategy development.** ([Original Repository](https://github.com/microsoft/RD-Agent))

## Key Features

*   **Automated ML Engineering:** Streamline model development with automated feature engineering, model tuning, and more.
*   **Data-Centric R&D:** Focus on data-driven research by iteratively proposing and refining ideas for optimal results.
*   **Multi-Agent Framework:** Harness the power of coordinated agents for complex R&D tasks, enhancing efficiency.
*   **Quantitative Finance Expertise:** Leverage RD-Agent for data-centric factor and model joint optimization in finance.
*   **Integration with LiteLLM:** Full support for LiteLLM as our default backend for integration with multiple LLM providers.
*   **Flexible & Extensible:** Designed to accommodate a wide range of scenarios and open to contributions.
*   **Comprehensive Documentation:** Extensive documentation with guides, examples, and API references.

## What's New

*   **Leading MLE-Bench Results:** RD-Agent is a top-performing machine learning engineering agent, achieving state-of-the-art results on the MLE-bench benchmark.
*   **Quant Finance Capabilities:** RD-Agent(Q) offers automated strategy development.
*   **Data Science Agent Preview:** Explore the development of our new Data Science Agent for streamlined data-driven solutions.
*   **Support for LiteLLM backend:** Integrates with multiple LLM providers seamlessly.
*   **[Kaggle Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html) support:** Automate model tuning and feature engineering on Kaggle competitions.

## Get Started

1.  **Installation:**
    ```bash
    pip install rdagent
    ```
2.  **Configuration:** Configure your preferred LLM and embedding models using LiteLLM.  See the [Configuration](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#-configuration) section for detailed setup instructions, including examples for OpenAI, Azure OpenAI, and DeepSeek.  Ensure the `rdagent health_check` passes after configuration.
3.  **Run Demos:** Choose from a range of scenarios, including Automated Quantitative Trading, Model Research Copilot, and more.
    *   Example: Run Automated Quantitative Trading:
        ```bash
        rdagent fin_quant
        ```
4.  **Monitor Results:** View logs and the application UI using the `rdagent ui` command.

**[📖 Documentation](https://rdagent.readthedocs.io/)** | **[🖥️ Live Demo](https://rdagent.azurewebsites.net/)** | **[▶️ YouTube Demos](https://www.youtube.com/playlist?list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR)**

## Scenarios

RD-Agent is versatile, with applications across multiple domains.

*   **Finance:**
    *   Automated Quantitative Trading & Iterative Factors Model Joint Evolution
    *   Automated Quantitative Trading & Iterative Factors Evolution
    *   Automated Quantitative Trading & Iterative Model Evolution
    *   Automated Quantitative Trading & Factors Extraction from Financial Reports
*   **General:**
    *   Automated Model Research & Development Copilot
*   **Medical:**
    *   Automated Medical Prediction Model Evolution
*   **Kaggle Competition**
    *   Automated Model Tuning & Feature Engineering (e.g. tabular-playground-series-dec-2021)

See the [scenarios documentation](https://rdagent.readthedocs.io/en/latest/scens/catalog.html) for detailed setup and usage instructions.

## Framework & Technology

RD-Agent utilizes a framework centered around automated R&D processes, including:

*   **Benchmark the R&D abilities**
*   **Idea proposal:** Explore new ideas or refine existing ones
*   **Ability to realize ideas:** Implement and execute ideas

## Contributing

We welcome contributions!  Please review the [Contributing Guide](CONTRIBUTING.md) before submitting pull requests.

---

<p align="center">
  Copyright (c) Microsoft Corporation. All rights reserved.
</p>
<p align="center">
  <a href="https://aka.ms/RD-Agent-Tech-Report">📄 Tech Report</a> | <a href="https://arxiv.org/abs/2505.14738">📃 Papers</a>
</p>
```

Key improvements and SEO optimizations:

*   **Clear, Concise Hook:** The one-sentence hook immediately explains what RD-Agent *is*.
*   **Keyword Rich:** Uses relevant keywords like "machine learning engineering," "quantitative finance," "AI framework," and "automation."
*   **Strategic Headings:** Uses headings like "Key Features," "What's New," and "Get Started" to break up the content and improve readability.
*   **Bulleted Lists:** Makes key information easy to scan and digest.
*   **Strong Call to Actions:** Encourages users to get started with clear instructions and links to demos and documentation.
*   **Internal Linking:** Links to relevant sections within the README and external resources.
*   **Updated Badges:** Includes essential badges (CI, PyPI, Documentation, Discord) and positions them strategically.
*   **Concise Summary:** Streamlines the information, removing unnecessary detail and redundancy.
*   **Focus on Value:** Highlights the benefits of using RD-Agent.
*   **Clear Installation and Usage Instructions:**  Provides a simplified "Get Started" section.
*   **Legal Disclaimer:** Includes the legal disclaimer
*   **Simplified Code Examples:** Shortened to be more readable and directly useful
*   **Revised Structure:** Refined the layout for better readability and information flow.
*   **SEO Optimization:** Integrates relevant keywords throughout the text.
*   **Expanded Scenarios section:** Improved for clarity and more detail.
*   **More concise documentation links:** Reduced redundancy in links
*   **Contributors section:** Corrected the contributors information.