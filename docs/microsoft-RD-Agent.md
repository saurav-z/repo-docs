# RD-Agent: Automate Machine Learning Engineering with AI 🚀

**RD-Agent is an innovative AI framework designed to automate and accelerate the machine learning engineering process, leading the charge in the [MLE-bench](https://github.com/openai/mle-bench) benchmark.**

[🖥️ Live Demo](https://rdagent.azurewebsites.net/) | [🎥 Demo Video](https://rdagent.azurewebsites.net/factor_loop) | [📖 Documentation](https://rdagent.readthedocs.io/en/latest/index.html) | [📄 Tech Report](https://aka.ms/RD-Agent-Tech-Report) | [Original Repository](https://github.com/microsoft/RD-Agent)

---

## ✨ Key Features

*   **Automated R&D:** Streamlines the entire machine learning engineering workflow, from idea generation to implementation and evaluation.
*   **Multi-Agent Framework:** Leverages a collaborative multi-agent system to tackle complex tasks, mirroring real-world R&D processes.
*   **Data-Centric Approach:** Focuses on data-driven scenarios, optimizing model development and data building.
*   **Top-Performing Agent:** Currently leads on the [MLE-bench](https://github.com/openai/mle-bench), demonstrating state-of-the-art performance in ML engineering tasks.
*   **Modular and Extensible:** Designed for flexibility and scalability, allowing for easy integration of new methods and scenarios.
*   **Financial Agent:** Provides a multi-agent framework for data-centric factors and model joint optimization.

## 🏆 Leading Performance on MLE-Bench

RD-Agent sets a new standard for ML engineering automation, achieving top results on the MLE-bench benchmark:

| Agent                   | Low == Lite (%) | Medium (%) | High (%) | All (%) |
| ----------------------- | --------------- | ---------- | -------- | ------- |
| R&D-Agent o1-preview    | 48.18 ± 2.49    | 8.95 ± 2.36  | 18.67 ± 2.98 | 22.4 ± 1.1   |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21    | 7.89 ± 3.33  | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview         | 34.3 ± 2.4      | 8.8 ± 1.1    | 10.0 ± 1.9  | 16.9 ± 1.1  |

**Learn more about these results:**

*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

## 🚀 Quick Start

### 🛠️ Installation

1.  **Prerequisites:** Requires Linux and Docker.  Ensure Docker is installed.
2.  **Conda Environment:** Create and activate a Conda environment:
    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```
3.  **Install RD-Agent:**
    *   **Using PyPI:** `pip install rdagent`
    *   **From Source (for developers):**
        ```bash
        git clone https://github.com/microsoft/RD-Agent
        cd RD-Agent
        make dev
        ```

### ⚙️ Configuration

Configure your LLM and Embedding models.  We recommend using LiteLLM, with support for various providers. Configuration instructions are available within the original [README](https://github.com/microsoft/RD-Agent).

### 🚀 Run an Example
```bash
rdagent fin_quant
```
See the [scenarios documentation](https://rdagent.readthedocs.io/en/latest/scens/catalog.html) to view more demos.

### 🖥️ Monitor Results

Run the following command to view logs:
```bash
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

## 🏭 Scenarios & Applications

RD-Agent is designed for various data-driven R&D scenarios:

*   **Finance:** Automate quantitative trading strategy development and factor optimization.
*   **Medical:** Develop medical prediction models.
*   **General:** Automate paper reading and model implementation, along with automated Kaggle model tuning and feature engineering.

### 📰 News

*   **[Technical Report Release](https://arxiv.org/abs/2505.14738):** Overall framework description and MLE-bench results.
*   **[R&D-Agent-Quant Release](https://arxiv.org/abs/2505.15155):** Application of R&D-Agent to quant trading.
*   **MLE-Bench Results Released:** RD-Agent currently leads on MLE-bench.
*   **LiteLLM Support:**  Full support for [LiteLLM](https://github.com/BerriAI/litellm) as a backend.
*   **Data Science Agent:** [Data Science Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)
*   **Kaggle Scenario:** [Kaggle Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)
*   **Community:** Official [Discord](https://discord.gg/ybQ97B6Jjy)
*   **More News**  [See the original README](https://github.com/microsoft/RD-Agent)

## 🤝 Contributing

Contributions are welcome! Refer to the [Contributing Guide](CONTRIBUTING.md) in the original repository.

## ⚖️ Legal

See the [Legal disclaimer](https://github.com/microsoft/RD-Agent#-%EF%B8%8F-legal-disclaimer) in the original README.

---

**[Explore the full potential of RD-Agent at the original repository!](https://github.com/microsoft/RD-Agent)**