<!-- Improved README.md for RD-Agent -->

<h1 align="center">RD-Agent: Automate Your Data-Driven R&D with AI</h1>

<div align="center">
  <a href="https://github.com/microsoft/RD-Agent" target="_blank">
    <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:40%;">
  </a>
</div>

**RD-Agent is an innovative AI framework designed to automate and accelerate the research and development process in data-driven fields like machine learning and quantitative finance.**  Learn more about the project at the [original repo](https://github.com/microsoft/RD-Agent).

---

## 🚀 Key Features

*   **AI-Powered Automation:** Automates critical aspects of R&D, from idea generation to implementation.
*   **Multi-Agent Framework:** Employs a collaborative approach, breaking down complex tasks into specialized agents.
*   **Data-Centric Focus:** Prioritizes data exploration, feature engineering, and model optimization.
*   **Real-World Applications:** Demonstrated success in machine learning engineering (MLE-bench) and quantitative finance.
*   **Flexible Scenarios:** Supports various applications, including model building, data mining, and Kaggle competitions.

---

## 🌟 What's New

Stay up-to-date on the latest features and developments:

*   **Top Performer on MLE-Bench:**  RD-Agent leads as the top-performing machine learning engineering agent, showcasing its capabilities in real-world ML engineering scenarios.
*   **R&D-Agent-Quant Release:** Applied R&D-Agent to quant trading, showcasing its application in financial markets.
*   **LiteLLM Support:** Improved support for LLM providers.
*   **Data Science Agent:** Explores scenarios, including medical predictions and Kaggle competitions.
*   **Expanded Documentation:**  Comprehensive documentation is available to support users.

---

## 📊 Performance Highlights

RD-Agent consistently delivers impressive results in benchmark tests and real-world applications:

### MLE-Bench Performance

RD-Agent demonstrates superior performance on the MLE-bench benchmark, outperforming previous solutions.

| Agent | Low == Lite (%) | Medium (%) | High (%) | All (%) |
|---------|--------|-----------|---------|----------|
| R&D-Agent o1-preview | 48.18 ± 2.49 | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21 | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview | 34.3 ± 2.4 | 8.8 ± 1.1 | 10.0 ± 1.9 | 16.9 ± 1.1 |

View the detailed runs for [R&D-Agent o1-preview](https://aka.ms/RD-Agent_MLE-Bench_O1-preview) and [R&D-Agent o3(R)+GPT-4.1(D)](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41).

---

## ⚙️ Quick Start

Get up and running with RD-Agent quickly:

### Prerequisites

*   **Operating System:** Linux (Docker recommended).
*   **Python:** 3.10 or 3.11
*   **Docker**

### Installation

1.  **Create a Conda Environment:**
    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```
2.  **Install RD-Agent:**
    ```bash
    pip install rdagent
    ```
    For developers, install from source using `make dev` after cloning the repo.

### Configuration
Set your Chat Model and Embedding Model in the following ways:
*   **Using LiteLLM (Default)**
    ```bash
    cat << EOF  > .env
    # Set to any model supported by LiteLLM.
    CHAT_MODEL=gpt-4o 
    EMBEDDING_MODEL=text-embedding-3-small
    # Configure unified API base
    OPENAI_API_BASE=<your_unified_api_base>
    OPENAI_API_KEY=<replace_with_your_openai_api_key>
    ```
    Other configuration options include:
    *   Azure OpenAI Setup
    *   DeepSeek Setup

### Run the Application

Choose your scenario and run the corresponding command:

*   Automated Quantitative Trading & Iterative Factors Model Joint Evolution:
    ```bash
    rdagent fin_quant
    ```
*   Automated Quantitative Trading & Iterative Factors Evolution:
    ```bash
    rdagent fin_factor
    ```
*   Automated Quantitative Trading & Iterative Model Evolution:
    ```bash
    rdagent fin_model
    ```
*   Automated Quantitative Trading & Factors Extraction from Financial Reports:
    ```bash
    rdagent fin_factor_report --report_folder=<Your financial reports folder path>
    ```
*   Automated Model Research & Development Copilot:
    ```bash
    rdagent general_model <Your paper URL>
    ```
*   Automated Medical Prediction Model Evolution:
    ```bash
    rdagent data_science --competition arf-12-hours-prediction-task
    ```
*   Automated Kaggle Model Tuning & Feature Engineering:
    ```bash
    rdagent data_science --competition tabular-playground-series-dec-2021
    ```

### Monitor Results

Use the UI to track progress and view logs:

```bash
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

---

## 📚 Documentation and Resources

*   **[🖥️ Live Demo](https://rdagent.azurewebsites.net/)**: Explore interactive demos and examples.
*   **[📖 Documentation](https://rdagent.readthedocs.io/en/latest/index.html)**:  Comprehensive documentation.
*   **[📄 Tech Report](https://aka.ms/RD-Agent-Tech-Report)**: Dive deeper into the technical details.
*   **[🎥 Demo Video](https://rdagent.azurewebsites.net/factor_loop)**: Watch a demonstration.
*   **[▶️ YouTube](https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR)**:  Check out videos on YouTube.
*   **[arXiv](https://arxiv.org/abs/2505.14738)**: Read the R&D-Agent paper.

---

## 🤝 Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) for details.

<div align="center">
  <img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>
</div>

<div align="center">
<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>
</div>

---

## ⚖️ Legal Disclaimer
*   The RD-agent is provided “as is”, without warranty of any kind. Users shall independently assess and test the risks of the RD-agent in a specific use scenario, ensure the responsible use of AI technology, including but not limited to developing and integrating risk mitigation measures, and comply with all applicable laws and regulations in all applicable jurisdictions. The RD-agent does not provide financial opinions or reflect the opinions of Microsoft. The inputs and outputs of the RD-agent belong to the users and users shall assume all liability under any theory of liability.