<h1 align="center">🤖 RD-Agent: Automating Data-Driven Research and Development</h1>

<p align="center">
  <a href="https://github.com/microsoft/RD-Agent">
    <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:40%; margin-bottom:10px;">
  </a>
</p>

<p align="center">
  <em>RD-Agent is a cutting-edge, multi-agent framework designed to automate the entire data-driven research and development lifecycle, leading the charge in AI-powered innovation.</em>
  <br>
  <a href="https://github.com/microsoft/RD-Agent">Explore the RD-Agent Repository</a>
</p>

<div align="center">
  <a href="https://rdagent.azurewebsites.net" target="_blank">🖥️ Live Demo</a> |
  <a href="https://rdagent.azurewebsites.net/factor_loop" target="_blank">🎥 Demo Video</a> |
  <a href="https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR" target="_blank">▶️YouTube</a> |
  <a href="https://rdagent.readthedocs.io/en/latest/index.html" target="_blank">📖 Documentation</a> |
  <a href="https://aka.ms/RD-Agent-Tech-Report" target="_blank">📄 Tech Report</a> |
  <a href="#-paperwork-list"> 📃 Papers </a>
  | <a href="https://discord.gg/ybQ97B6Jjy"> 💬 Discord </a>
</div>

[![CI](https://github.com/microsoft/RD-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/ci.yml)
[![CodeQL](https://github.com/microsoft/RD-Agent/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/github-code-scanning/codeql)
[![Dependabot Updates](https://github.com/microsoft/RD-Agent/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/dependabot/dependabot-updates)
[![Lint PR Title](https://github.com/microsoft/RD-Agent/actions/workflows/pr.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/pr.yml)
[![Release.yml](https://github.com/microsoft/RD-Agent/actions/workflows/release.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/release.yml)
[![Platform](https://img.shields.io/badge/platform-Linux-blue)](https://pypi.org/project/rdagent/#files)
[![PyPI](https://img.shields.io/pypi/v/rdagent)](https://pypi.org/project/rdagent/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rdagent)](https://pypi.org/project/rdagent/)
[![Release](https://img.shields.io/github/v/release/microsoft/RD-Agent)](https://github.com/microsoft/RD-Agent/releases)
[![GitHub](https://img.shields.io/github/license/microsoft/RD-Agent)](https://github.com/microsoft/RD-Agent/blob/main/LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation Status](https://readthedocs.org/projects/rdagent/badge/?version=latest)](https://rdagent.readthedocs.io/en/latest/?badge=latest)
[![Readthedocs Preview](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2505.14738-00ff00.svg)](https://arxiv.org/abs/2505.14738)

## Key Features

*   **Automated R&D:** Automates critical R&D processes, from idea generation to implementation.
*   **Multi-Agent Framework:**  Employs a collaborative multi-agent system for comprehensive task execution.
*   **Data-Centric Approach:** Focuses on data-driven scenarios, enhancing model and data development.
*   **Integration with MLE-bench:** Demonstrates leadership in machine learning engineering tasks on the MLE-bench benchmark.
*   **Quant Finance Capabilities:**  Features RD-Agent(Q), a specialized framework for automating quantitative finance strategy development.
*   **Modular Design:**  Supports flexible integration with various LLM providers via LiteLLM.
*   **Active Development:**  Continuously enhanced with new features and scenarios to boost R&D productivity.

## 🥇 MLE-bench Leader

RD-Agent is currently the leading machine learning engineering agent on the MLE-bench benchmark, showcasing significant advancements in automating complex ML tasks.

| Agent                    | Low == Lite (%) | Medium (%) | High (%) | All (%)   |
| ------------------------ | --------------- | ---------- | -------- | --------- |
| R&D-Agent o1-preview     | 48.18 ± 2.49    | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1  |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21    | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview          | 34.3 ± 2.4      | 8.8 ± 1.1   | 10.0 ± 1.9 | 16.9 ± 1.1 |

**Explore the Results:**
*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

## 🥇 RD-Agent(Q): Revolutionizing Quantitative Finance

RD-Agent(Q) is the first data-centric, multi-agent framework designed to automate the full-stack research and development of quantitative strategies.

*   **Exceptional Performance:** Achieves up to 2x higher ARR compared to benchmark factor libraries.
*   **Efficient Factor Utilization:** Uses over 70% fewer factors than traditional methods.
*   **Superior to Deep Time-Series Models:** Outperforms state-of-the-art models with smaller resource budgets.
*   **Robust Strategy Development:** Delivers an excellent trade-off between predictive accuracy and strategy robustness.

**Learn More:**  [RD-Agent(Q) Paper](https://arxiv.org/abs/2505.15155) and [Documentation](https://rdagent.readthedocs.io/en/latest/scens/quant_agent_fin.html)

## 📰 Recent News & Updates

*   **[Technical Report Release](#overall-technical-report):** Overall framework description and results on MLE-bench
*   **[R&D-Agent-Quant Release](#deep-application-in-diverse-scenarios):** Apply R&D-Agent to quant trading
*   **MLE-Bench Results Released:** RD-Agent leads as the [top-performing machine learning engineering agent](#-the-best-machine-learning-engineering-agent) on MLE-bench
*   **LiteLLM Integration:** Full support for [LiteLLM](https://github.com/BerriAI/litellm) for versatile LLM provider integration.
*   **Data Science Agent & Kaggle Agent:** Explore our [Data Science Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html) and [Kaggle Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)
*   **Community Channels:** Join our [Discord](https://discord.gg/ybQ97B6Jjy) for real-time discussion and support.

## 🚀 Quick Start

### Prerequisites

*   **Operating System:** RD-Agent currently only supports Linux.
*   **Docker:** Ensure Docker is installed (see [Docker Installation](https://docs.docker.com/engine/install/)). Verify installation with `docker run hello-world`.
*   **Conda Environment:** Create and activate a Conda environment:
    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```

### Installation

1.  **Install from PyPI:**
    ```bash
    pip install rdagent
    ```
2.  **Install from Source (for development):**
    ```bash
    git clone https://github.com/microsoft/RD-Agent
    cd RD-Agent
    make dev
    ```
    More details can be found in the [development setup](https://rdagent.readthedocs.io/en/latest/development.html).

### Health Check

Verify your setup:
```bash
rdagent health_check --no-check-env
```

### Configuration

Configure your Chat and Embedding Models. The default setup uses LiteLLM for easy integration with multiple LLM providers.

**Option 1: Unified API base for both models**
```bash
cat << EOF  > .env
# Set to any model supported by LiteLLM.
CHAT_MODEL=gpt-4o 
EMBEDDING_MODEL=text-embedding-3-small
# Configure unified API base
OPENAI_API_BASE=<your_unified_api_base>
OPENAI_API_KEY=<replace_with_your_openai_api_key>
```
**Option 2: Separate API bases for Chat and Embedding models**
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
**DeepSeek Setup Example:**
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
**Health Check Again**
```bash
rdagent health_check
```
### 🚀 Run the Application
Run a demo by executing one of the following commands:
*   **Automated Quantitative Trading & Iterative Factors Model Joint Evolution:**
    ```sh
    rdagent fin_quant
    ```
*   **Automated Quantitative Trading & Iterative Factors Evolution:**
    ```sh
    rdagent fin_factor
    ```
*   **Automated Quantitative Trading & Iterative Model Evolution:**
    ```sh
    rdagent fin_model
    ```
*   **Automated Quantitative Trading & Factors Extraction from Financial Reports:**
    ```sh
    rdagent fin_factor_report --report_folder=<Your financial reports folder path>
    ```
*   **Automated Model Research & Development Copilot:**
    ```sh
    rdagent general_model <Your paper URL>
    # Example: rdagent general_model "https://arxiv.org/pdf/2210.09789"
    ```
*   **Automated Medical Prediction Model Evolution:**
    ```bash
    rdagent data_science --competition arf-12-hours-prediction-task
    ```
*   **Automated Kaggle Model Tuning & Feature Engineering:**
    ```bash
    rdagent data_science --competition tabular-playground-series-dec-2021
    ```

### 🖥️ Monitor Results
View application logs with:

```sh
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

## 🏭 Scenarios & Demos

RD-Agent is applied to various data-driven scenarios, showcasing its versatility:

| Scenario/Target | Model Implementation                   | Data Building                                                                      |
| --------------- | -------------------------------------- | ----------------------------------------------------------------------------------- |
| **💹 Finance**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop) <br/> [▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s)   |  🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) <br/> [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s) <br/>   🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor) <br/> [▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)  |
| **🩺 Medical**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm) <br/> [▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4) | -                                                                                  |
| **🏭 General**      | 🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model) <br/> [▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKQ7o) <br/> 🤖 Auto Kaggle Model Tuning   | 🤖Auto Kaggle feature Engineering |

## ⚙️ Framework Overview

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

The RD-Agent framework is designed to automate and evolve the R&D process in data science.

*   **Key Research Areas:** Benchmark, Idea Proposal, and Implementation.
*   **Core Goal:** Build an agent that continuously learns and improves its R&D skills.

More details can be found in the **[📖 readthedocs](https://rdagent.readthedocs.io/)**.

## 📃 Paper/Work List

### Overall Technical Report
-   [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)

### 📊 Benchmark
-   [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)

### 🔍 Research
RD-Agent automates the R&D cycle, including hypothesis generation, experiment design, implementation, and feedback analysis.

### 🛠️ Development
-   [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)

### Deep Application in Diverse Scenarios
-   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

## 🤝 Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) for details.

<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

## ⚖️ Legal Disclaimer

The RD-Agent is provided "as is," without warranty. Users are responsible for risk assessment, compliance, and adherence to all laws and regulations. The RD-Agent does not provide financial advice or substitute financial professionals. Users are liable for their use of the RD-Agent and its outputs.