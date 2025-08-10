# RD-Agent: Automate Your Data-Driven R&D with AI

**Unlock the future of machine learning engineering!** RD-Agent is a cutting-edge, multi-agent framework developed by Microsoft to automate and accelerate the entire data-driven R&D process.  Explore the power of AI-driven innovation and streamline your workflows.

[**View the Live Demo**](https://rdagent.azurewebsites.net/) | [**Explore the Documentation**](https://rdagent.readthedocs.io/en/latest/index.html) | [**Check Out the GitHub Repository**](https://github.com/microsoft/RD-Agent)

## Key Features

*   **Automated R&D Process:** Automates the critical stages of machine learning engineering, from idea generation to implementation.
*   **Multi-Agent Framework:** Leverages coordinated agents for research and development, facilitating collaborative and efficient workflows.
*   **Data-Centric Approach:** Focuses on automating model building and data discovery, improving results.
*   **Versatile Applications:** Applicable to a broad range of data-driven scenarios, including finance, medical research, and more.
*   **Proven Performance:** Achieves state-of-the-art results on MLE-bench and in real-world financial markets.
*   **Easy to Deploy:** Supports Docker and Conda for straightforward setup and usage.
*   **Modular and Extensible:** Designed to accommodate additional methods and scenarios.

## 🥇 The Best Machine Learning Engineering Agent on MLE-bench!

R&D-Agent is at the forefront of AI-driven machine learning engineering, showcasing remarkable performance on the MLE-bench benchmark.

| Agent                     | Low == Lite (%) | Medium (%) | High (%) | All (%)  |
| :------------------------ | :-------------: | :--------: | :-------: | :-------: |
| R&D-Agent o1-preview      |  48.18 ± 2.49  | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) |  51.52 ± 6.21  | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview           |   34.3 ± 2.4   |  8.8 ± 1.1 |  10.0 ± 1.9 |  16.9 ± 1.1 |

*   **O3(R)+GPT-4.1(D):** Integrates Research Agent (o3) with Development Agent (GPT-4.1) for cost-effective and efficient performance.
*   **AIDE o1-preview:** The previously best public result on MLE-bench.

See the detailed runs:

*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

## 🥇 R&D-Agent(Q): The First Data-Centric Quant Multi-Agent Framework!

RD-Agent for Quantitative Finance (RD-Agent(Q)) revolutionizes quantitative strategy development with its data-centric, multi-agent approach.

![RD-Agent(Q) Overview](https://github.com/user-attachments/assets/3198bc10-47ba-4ee0-8a8e-46d5ce44f45d)

Key Highlights:

*   **Significantly Higher ARR:** Achieves approximately 2x higher Annualized Rate of Return (ARR) than benchmark factor libraries.
*   **Reduced Factor Usage:** Employs over 70% fewer factors, streamlining strategy complexity.
*   **Superior Performance:** Outperforms state-of-the-art deep time-series models while requiring fewer resources.
*   **Optimized Trade-Off:** Offers an excellent balance between predictive accuracy and strategy robustness.

Explore RD-Agent(Q) further:

*   [Paper](https://arxiv.org/abs/2505.15155)
*   [Documentation](https://rdagent.readthedocs.io/en/latest/scens/quant_agent_fin.html)

## 📰 What's New

Stay up-to-date on the latest developments in RD-Agent:

*   **[Overall Technical Report Release](https://arxiv.org/abs/2505.14738):** Comprehensive framework description and MLE-bench results.
*   **[R&D-Agent-Quant Release](https://arxiv.org/abs/2505.15155):** Application of RD-Agent to quant trading.
*   **MLE-Bench Results Released:** RD-Agent currently leads as the [top-performing machine learning engineering agent](#-the-best-machine-learning-engineering-agent) on MLE-bench.
*   **LiteLLM Backend Support:** Enhanced integration with multiple LLM providers via LiteLLM.
*   **General Data Science Agent:** Explore the [Data Science Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html).
*   **Kaggle Scenario Release:** Try out the new [Kaggle Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html).
*   **Official Community Channels:** Join our [Discord](https://discord.gg/ybQ97B6Jjy) community for discussions and updates.

## 🚀 Quick Start

Get started with RD-Agent in three easy steps:

1.  **Prerequisites:** RD-Agent currently supports Linux. Make sure you have [Docker](https://docs.docker.com/engine/install/) installed and configured correctly.
2.  **Environment Setup:**
    *   Create a Conda environment: `conda create -n rdagent python=3.10`
    *   Activate the environment: `conda activate rdagent`
3.  **Installation:**
    *   Install the package:  `pip install rdagent`

### Configuration

Set your `CHAT_MODEL` and `EMBEDDING_MODEL` in your `.env` file using either a unified or separate API base:

**Example using LiteLLM (recommended):**

**Unified API base for both models:**

*Configuration Example: `OpenAI` Setup :*

```bash
cat << EOF  > .env
# Set to any model supported by LiteLLM.
CHAT_MODEL=gpt-4o 
EMBEDDING_MODEL=text-embedding-3-small
# Configure unified API base
OPENAI_API_BASE=<your_unified_api_base>
OPENAI_API_KEY=<replace_with_your_openai_api_key>
```

*Configuration Example: `Azure OpenAI` Setup :*

> Before using this configuration, please confirm in advance that your `Azure OpenAI API key` supports `embedded models`.

```bash
cat << EOF  > .env
EMBEDDING_MODEL=azure/<Model deployment supporting embedding>
CHAT_MODEL=azure/<your deployment name>
AZURE_API_KEY=<replace_with_your_openai_api_key>
AZURE_API_BASE=<your_unified_api_base>
AZURE_API_VERSION=<azure api version>
```

**Separate API bases for Chat and Embedding models**
```bash
cat << EOF  > .env
# Set to any model supported by LiteLLM.
# Configure separate API bases for chat and embedding
  
# CHAT MODEL:
CHAT_MODEL=gpt-4o 
OPENAI_API_BASE=<your_chat_api_base>
OPENAI_API_KEY=<replace_with_your_openai_api_key>

# EMBEDDING MODEL:
# TAKE siliconflow as an example, you can use other providers.
# Note: embedding requires litellm_proxy prefix
EMBEDDING_MODEL=litellm_proxy/BAAI/bge-large-en-v1.5
LITELLM_PROXY_API_KEY=<replace_with_your_siliconflow_api_key>
LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1
```

**DeepSeek Configuration:**

```bash
cat << EOF  > .env
# CHAT MODEL: Using DeepSeek Official API
CHAT_MODEL=deepseek/deepseek-chat 
DEEPSEEK_API_KEY=<replace_with_your_deepseek_api_key>

# EMBEDDING MODEL: Using SiliconFlow for embedding since deepseek has no embedding model.
# Note: embedding requires litellm_proxy prefix
EMBEDDING_MODEL=litellm_proxy/BAAI/bge-m3
LITELLM_PROXY_API_KEY=<replace_with_your_siliconflow_api_key>
LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1
```
### Health Check

Verify your setup with: `rdagent health_check`

### Run the Application

*   **Automated Quantitative Trading & Iterative Factors Model Joint Evolution:**  `rdagent fin_quant`
*   **Automated Quantitative Trading & Iterative Factors Evolution:**  `rdagent fin_factor`
*   **Automated Quantitative Trading & Iterative Model Evolution:**  `rdagent fin_model`
*   **Automated Quantitative Trading & Factors Extraction from Financial Reports:** `rdagent fin_factor_report --report_folder=<Your financial reports folder path>`
*   **Automated Model Research & Development Copilot:** `rdagent general_model  "https://arxiv.org/pdf/2210.09789"`
*   **Automated Medical Prediction Model Evolution:** `rdagent data_science --competition arf-12-hours-prediction-task`
*   **Automated Kaggle Model Tuning & Feature Engineering:** `rdagent data_science --competition tabular-playground-series-dec-2021`

### Monitor Results
```sh
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

## 🏭 Scenarios

RD-Agent is applied to several key data-driven industrial scenarios:

| Scenario/Target | Model Implementation                   | Data Building                                                                      |
| :-------------- | :------------------------------------- | :----------------------------------------------------------------------------------- |
| **💹 Finance**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop) |  🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) <br/>   🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)  |
| **🩺 Medical**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm) | -                                                                                  |
| **🏭 General**      | 🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model) <br/> 🤖 Auto Kaggle Model Tuning   | 🤖Auto Kaggle feature Engineering |

## ⚙️ Framework Overview

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

The framework focuses on:

*   **Benchmarking:** Evaluating R&D abilities.
*   **Idea Proposal:** Exploring new ideas and refining existing ones.
*   **Idea Realization:** Implementing and executing ideas.

## 📃 Papers & Resources

*   **[Overall Technical Report](https://arxiv.org/abs/2505.14738)**
*   **[Benchmark Paper](https://arxiv.org/abs/2404.11276)**
*   **[Research Paper](https://rdagent.azurewebsites.net)**
*   **[Development Paper](https://arxiv.org/abs/2407.18690)**
*   **[R&D-Agent-Quant Paper](https://arxiv.org/abs/2505.15155)**

## 🤝 Contributing

We welcome contributions!  Check the [Contributing Guide](CONTRIBUTING.md) for details.

<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

## ⚖️ Legal Disclaimer

Please refer to the [Legal disclaimer](https://github.com/microsoft/RD-Agent#-%EF%B8%8F-legal-disclaimer) section for important information regarding the use of RD-Agent.