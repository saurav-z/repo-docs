<h1 align="center">RD-Agent: Automate Your Data-Driven R&D with AI</h1>

<p align="center">
    RD-Agent empowers you to automate and accelerate your research and development processes using cutting-edge AI, leading the way in machine learning engineering. Explore the future of R&D today!  <a href="https://github.com/microsoft/RD-Agent">Explore the RD-Agent Repository</a>.
</p>

---

## Key Features

*   **Leading Performance:** Top-performing machine learning engineering agent on the MLE-bench, achieving impressive results across various complexity levels.
*   **Data-Centric Approach:**  A novel, data-centric framework designed for automated R&D, with a focus on continuous improvement.
*   **Multi-Agent Framework:**  Leverages a multi-agent architecture for coordinated factor-model co-optimization, leading to superior results in quantitative finance and beyond.
*   **Versatile Scenarios:** Supports a wide range of applications, including automated quantitative trading, model research, and Kaggle competition participation.
*   **Easy to Use:**  Offers a straightforward setup with Docker and Conda, making it accessible for both developers and researchers.
*   **Demo & Live Access:**  Get hands-on experience with live demos and a user-friendly interface to visualize your results.
*   **Extensive Documentation:**  Comprehensive documentation and a dedicated community provide support for implementation and customization.

---

## About RD-Agent

RD-Agent is a cutting-edge, AI-driven framework designed to revolutionize the data-driven R&D process. It focuses on automating the most critical aspects of model and data development, leading to increased efficiency and improved outcomes.

### 🥇 Machine Learning Engineering Agent on MLE-Bench

RD-Agent consistently demonstrates high performance, outperforming existing solutions on the [MLE-bench](https://github.com/openai/mle-bench).

**Performance Highlights:**

| Agent                      | Low == Lite (%) | Medium (%) | High (%) | All (%) |
| -------------------------- | --------------- | ----------- | -------- | -------- |
| R&D-Agent o1-preview       | 48.18 ± 2.49    | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1  |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21    | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview            | 34.3 ± 2.4      | 8.8 ± 1.1   | 10.0 ± 1.9 | 16.9 ± 1.1 |

See detailed runs:
*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

### 🥇 Data-Centric Quant Multi-Agent Framework

RD-Agent is the pioneering data-centric, multi-agent framework designed to automate research and development of quantitative strategies via coordinated factor-model co-optimization, revolutionizing quant finance.

![RD-Agent Architecture](https://github.com/user-attachments/assets/3198bc10-47ba-4ee0-8a8e-46d5ce44f45d)

#### Highlights

*   **Superior Performance:**  Achieves a 2x higher ARR than benchmark factor libraries while using 70% fewer factors in real stock market experiments.
*   **Efficiency:**  Outperforms state-of-the-art deep time-series models at smaller resource budgets.
*   **Robustness:** Alternating factor-model optimization provides an excellent trade-off between predictive accuracy and strategy robustness.

See more details in the [paper](https://arxiv.org/abs/2505.15155) and reproduce through the [documentation](https://rdagent.readthedocs.io/en/latest/scens/quant_agent_fin.html).

## 📰 News and Updates

*   **[Overall Technical Report Release](#overall-technical-report):** Overall framework description and results on MLE-bench
*   **[R&D-Agent-Quant Release](#deep-application-in-diverse-scenarios):** Applying R&D-Agent to quant trading.
*   **MLE-Bench Results Released:** RD-Agent leading as top performer on MLE-bench.
*   **LiteLLM Backend Support:**  Full support for [LiteLLM](https://github.com/BerriAI/litellm) integration with multiple LLM providers.
*   **General Data Science Agent:** Explore the Data Science Agent scenario.
*   **Kaggle Scenario Release:**  Try the [Kaggle Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html) and leverage the new features.
*   **Official Discord:** Join the discussion on the [Discord](https://discord.gg/ybQ97B6Jjy).
*   **First Release:** The public debut of **R&D-Agent** on GitHub.

## 🚀 Get Started

1.  **Installation:** Follow the simple installation instructions using Docker and Conda.
2.  **Configuration:**  Configure your Chat and Embedding models. LiteLLM is now the default backend.
3.  **Run Demos:** Experience the power of RD-Agent by running the provided scenarios.

### 🛠️ Installation

*   **Prerequisites:** Linux, Docker, Conda.
*   **Install RD-Agent:** `pip install rdagent`
*   **Health Check:** Ensure your environment is configured correctly.
    ```bash
    rdagent health_check --no-check-env
    ```

### ⚙️ Configuration

*   Set up your `CHAT_MODEL` and `EMBEDDING_MODEL` using environment variables via LiteLLM. Examples include: OpenAI, Azure OpenAI, and DeepSeek setups.
*   See more details in the [documentation](https://rdagent.readthedocs.io/en/latest/installation_and_configuration.html).

### 🚀 Run the Application

```sh
# Run Automated Quantitative Trading
rdagent fin_quant

# Run Automated Model Research & Development Copilot
rdagent general_model "https://arxiv.org/pdf/2210.09789"

# ...and more, see the [scenarios catalog](https://rdagent.readthedocs.io/en/latest/scens/catalog.html)
```

### 🖥️ Monitor Results

```sh
rdagent ui --port 19899 --log_dir <your log folder> --data_science <True or False>
```

---

## 🏭 Scenarios

RD-Agent is applied to several valuable data-driven industrial scenarios.

*   **Finance:** Automate quantitative trading.
*   **Medical:**  Medical self-loop model proposal and implementation application
*   **General:**  Automate paper reading, model extraction and implementation.
*   **Kaggle:**  Automate model tuning and feature engineering for Kaggle competitions.

## 🤝 Contributing

We welcome contributions and suggestions!  Refer to the [Contributing Guide](CONTRIBUTING.md).

<img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>
<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

---

## 📃 Paper/Work List

### Overall Technical Report
*   [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)

### 📊 Benchmark
*   [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)

### 🔍 Research

*   In a data mining expert's daily research and development process, they propose a hypothesis, design experiments, implement the experiment as code, and execute the code to get feedback.

### 🛠️ Development
*   [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)

### Deep Application in Diverse Scenarios
*   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

---

## ⚖️ Legal Disclaimer

*   See the full disclaimer in the original README.

---

**[Go to Top](#rd-agent-automate-your-data-driven-rd-with-ai)**