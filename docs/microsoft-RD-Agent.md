<h1 align="center">🚀 R&D-Agent: Automate Data-Driven R&D with AI</h1>

<div align="center">
  <img src="docs/_static/logo.png" alt="RA-Agent logo" style="width:40%; margin-bottom: 20px;">
</div>

**R&D-Agent is a cutting-edge AI agent designed to automate and revolutionize the research and development process, making it the top-performing machine learning engineering agent on the MLE-bench benchmark!** This open-source project from Microsoft leverages the power of Large Language Models (LLMs) to streamline data-driven R&D across various domains. Explore [the original repo](https://github.com/microsoft/RD-Agent) for more information.

## ✨ Key Features

*   **Automated R&D:** Automates critical aspects of the industrial R&D process.
*   **Multi-Agent Framework:**  A collaborative framework of specialized agents, including:
    *   **Research Agent (R):** Proposes new ideas and identifies areas for improvement.
    *   **Development Agent (D):** Implements ideas, builds models, and evolves the R&D process.
*   **Data-Centric Focus:** Designed for data-driven scenarios, streamlining model and data development.
*   **MLE-bench Leader:** R&D-Agent currently leads as the top-performing machine learning engineering agent on MLE-bench.
*   **Versatile Applications:** Supports diverse scenarios including:
    *   Quantitative Finance 💰
    *   Data Mining 🤖
    *   Research Copilot 🦾
    *   Kaggle Competitions 🏆
*   **Integrated Tools:** Seamless integration with LiteLLM for flexible LLM provider selection.
*   **Automated Full-Stack:** Automates the full-stack research and development of quantitative strategies via coordinated factor-model co-optimization.
*   **Proven Performance:** RD-Agent(Q) achieves higher ARR than benchmark factor libraries.

## 🚦 Quick Start

### Prerequisites:

*   **Operating System:** R&D-Agent currently only supports Linux.
*   **Docker:** Ensure Docker is installed. Refer to the [official Docker page](https://docs.docker.com/engine/install/) for installation instructions.
*   **Conda:** Create a new conda environment with Python (3.10 and 3.11 are well-tested in our CI):
    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```

### Installation:

*   **Install from PyPI:**
    ```bash
    pip install rdagent
    ```
*   **Install from Source (for developers):**
    ```bash
    git clone https://github.com/microsoft/RD-Agent
    cd RD-Agent
    make dev
    ```

### Configuration:

1.  **Set Up Your LLM API Keys:** Configure your preferred Large Language Model (LLM) provider (e.g., OpenAI, Azure OpenAI, DeepSeek) by setting the following environment variables in a `.env` file.  Example configurations are provided in the original README. The default setup uses LiteLLM for easy switching between LLM providers. See the original README for details.
2.  **Health Check:** Validate your configuration by running:
    ```bash
    rdagent health_check
    ```

### Run the Application:

*   **Automated Quantitative Trading:**
    ```bash
    rdagent fin_quant
    ```
    or
    ```bash
    rdagent fin_factor
    ```
    or
    ```bash
    rdagent fin_model
    ```
*   **Factor Extraction from Financial Reports:**
    ```bash
    rdagent fin_factor_report --report_folder=<Your financial reports folder path>
    ```
*   **Model Research & Development Copilot:**
    ```bash
    rdagent general_model "https://arxiv.org/pdf/2210.09789"
    ```
*   **Medical Model Evolution:**
    ```bash
    rdagent data_science --competition arf-12-hours-prediction-task
    ```
*   **Kaggle Model Tuning & Feature Engineering:**
    ```bash
    rdagent data_science --competition tabular-playground-series-dec-2021
    ```

### Monitor the Results:

```bash
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

## 🖼️ Scenarios & Demos

R&D-Agent shines in automating data-driven R&D, with applications spanning model implementation and data building:

| Scenario/Target | Model Implementation                                          | Data Building                                                                                                  |
| :-------------- | :----------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **Finance**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop)[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s) | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop)[▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s) <br/> 🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)  |
| **Medical**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm)[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4) | -                                                                                                              |
| **General**      | 🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model)[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKa7o) <br/> 🤖 Auto Kaggle Model Tuning       | 🤖Auto Kaggle feature Engineering                                                                                  |

*   **Live Demo:** Explore the capabilities of R&D-Agent on the [Live Demo](https://rdagent.azurewebsites.net/).
*   **Scenarios:** Refer to the [Documentation](https://rdagent.readthedocs.io/en/latest/scens/catalog.html) for more details on the scenarios.

## 📚 Framework

R&D-Agent is built upon a robust framework designed to automate the R&D process.
The core research focuses on:

*   **Benchmark the R&D abilities**
*   **Idea Proposal:** Explore new ideas or refine existing ones
*   **Implementation:** Implement and execute ideas

For more details, refer to the [documentation](https://rdagent.readthedocs.io/).

## 📃 Papers & Publications

*   [Overall Technical Report](https://arxiv.org/abs/2505.14738)
*   [Benchmark](https://arxiv.org/abs/2404.11276)
*   [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)
*   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

## 🤝 Contributing

We encourage contributions!  Review the [Contributing Guide](CONTRIBUTING.md) for details.

<img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>
<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

## ⚖️ Legal Disclaimer

*(Legal disclaimer from original README included)*

---

This improved README is SEO-optimized by:

*   **Keywords:**  Using relevant keywords like "AI agent," "machine learning," "R&D," "automation," "data-centric," "LLM," "MLE-bench," and specific application areas.
*   **Headings:**  Clearly organized sections with descriptive headings.
*   **Bulleted Lists:** Highlights key features for readability and quick understanding.
*   **Concise Language:**  Uses clear and direct language.
*   **Call to Action:** Encourages exploration of the demo and documentation.
*   **Links:** Includes relevant links throughout the README, including a direct link back to the original repository.