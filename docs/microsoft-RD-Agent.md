<!-- Improved and Summarized README for RD-Agent -->

<h1 align="center">RD-Agent: Automating Data-Driven Research and Development</h1>

<p align="center">
  <a href="https://github.com/microsoft/RD-Agent">
    <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:30%;">
  </a>
  <br>
  <b>RD-Agent empowers AI to automate and accelerate the R&D process, unlocking new possibilities in machine learning and beyond.</b>
</p>

<div align="center">
  <a href="https://rdagent.azurewebsites.net" target="_blank">🖥️ Live Demo</a> |
  <a href="https://rdagent.azurewebsites.net/factor_loop" target="_blank">🎥 Demo Video</a> |
  <a href="https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR" target="_blank">▶️ YouTube</a> |
  <a href="https://rdagent.readthedocs.io/en/latest/index.html" target="_blank">📖 Documentation</a> |
  <a href="https://aka.ms/RD-Agent-Tech-Report" target="_blank">📄 Tech Report</a> |
  <a href="#paperwork-list">📃 Papers</a>
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
[![Chat](https://img.shields.io/badge/chat-discord-blue)](https://discord.gg/ybQ97B6Jjy)
[![Documentation Status](https://readthedocs.org/projects/rdagent/badge/?version=latest)](https://rdagent.readthedocs.io/projects/rdagent/?badge=latest)
[![Readthedocs Preview](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2505.14738-00ff00.svg)](https://arxiv.org/abs/2505.14738)

---

## Key Features

*   **Automated R&D:** Automates the R&D process, from idea generation to implementation and evaluation.
*   **Multi-Agent Framework:** Leverages a multi-agent system to coordinate complex tasks and foster innovation.
*   **Data-Centric Approach:** Focuses on data-driven scenarios to streamline model and data development.
*   **Leading Performance:** Achieves state-of-the-art results on the MLE-bench benchmark and in quantitative finance applications.
*   **Extensible and Modular:** Supports various scenarios and continuous integration with new methods and applications.

---

## 🥇 Machine Learning Engineering Agent - Top Performer

R&D-Agent is a leading agent on the [MLE-bench](https://github.com/openai/mle-bench) benchmark, designed to evaluate AI agents on real-world machine learning engineering tasks.

### Key Benchmarks

| Agent                     | Low == Lite (%) | Medium (%) | High (%) | All (%) |
| ------------------------- | --------------- | ---------- | -------- | -------- |
| R&D-Agent o1-preview      | 48.18 ± 2.49    | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21    | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview           | 34.3 ± 2.4      | 8.8 ± 1.1   | 10.0 ± 1.9 | 16.9 ± 1.1 |

**Detailed Runs:**
*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

## 🥇 Data-Centric Quant Finance Framework

RD-Agent(Q) is the first data-centric, multi-agent framework designed to automate the full-stack research and development of quantitative strategies via coordinated factor-model co-optimization.  It demonstrates significant improvements over benchmark libraries, achieving higher ARR with fewer factors, and surpassing state-of-the-art time-series models.

![RD-Agent(Q) Architecture](https://github.com/user-attachments/assets/3198bc10-47ba-4ee0-8a8e-46d5ce44f45d)

*   Read the paper: [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)
*   Explore the documentation: [RD-Agent Documentation](https://rdagent.readthedocs.io/en/latest/scens/quant_agent_fin.html)

---

## 📰 What's New

*   **[Technical Report Release](#overall-technical-report):** Overall framework description and results on MLE-bench
*   **[R&D-Agent-Quant Release](#deep-application-in-diverse-scenarios):** Applying R&D-Agent to quant trading
*   **MLE-Bench Results Released:** RD-Agent is the top-performing machine learning engineering agent on MLE-bench
*   **LiteLLM Backend Support:** Fully support for LiteLLM as the default backend for integration with multiple LLM providers.
*   **Data Science Agent:** [Data Science Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)
*   **Kaggle Scenario:** [Kaggle Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)
*   **Community Channels:** Official WeChat group and Discord channel releases.
*   **Initial Release:** The project launch on GitHub.

---

## 🌟 Introduction

RD-Agent focuses on automating critical aspects of the R&D process, focusing on data-driven scenarios.  It employs a framework with two primary components: 'R' for proposing new ideas and 'D' for implementing them.  The project aims to drive industrial value through the autonomous evolution of R&D capabilities.

<div align="center">
  <img src="docs/_static/scen.png" alt="Focused Scenario" style="width:80%;">
</div>

### Applications

RD-Agent serves as:

*   💰 **Automatic Quant Factory** ([🎥Demo Video](https://rdagent.azurewebsites.net/factor_loop)|[▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s))
*   🤖 **Data Mining Agent:** ([🎥Demo Video 1](https://rdagent.azurewebsites.net/model_loop)|[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s)) ([🎥Demo Video 2](https://rdagent.azurewebsites.net/dmm)|[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4))
*   🦾 **Research Copilot:** ([🎥Demo Video](https://rdagent.azurewebsites.net/report_model)|[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKQ7o)) / ([🎥Demo Video](https://rdagent.azurewebsites.net/report_factor)|[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c))
*   🤖 **Kaggle Agent:** (Coming Soon)
*   ...

Explore the demos:  **[🖥️ Live Demo](https://rdagent.azurewebsites.net/)**.

<div align="center">
  <a href="https://rdagent.azurewebsites.net/" target="_blank">
    <img src="docs/_static/demo.png" alt="Watch the demo" width="80%">
  </a>
</div>

---

## ⚡ Quick Start

### Prerequisites

*   **Linux:**  RD-Agent currently supports Linux.
*   **Docker:**  Install Docker following [Docker's official instructions](https://docs.docker.com/engine/install/). Ensure the user can run Docker commands without `sudo`.
*   **Python:** Python 3.10 or 3.11 are well-tested.
*   **Conda:**

    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```

### Installation

*   **Install from PyPI:**

    ```bash
    pip install rdagent
    ```

*   **Install from Source (for development):**

    ```bash
    git clone https://github.com/microsoft/RD-Agent
    cd RD-Agent
    make dev
    ```

    See the [development setup documentation](https://rdagent.readthedocs.io/en/latest/development.html) for details.

### Health Check

```bash
rdagent health_check --no-check-env
```

### Configuration

*   **Chat and Embedding Models:**  Set your preferred models.

    > **Note:** Experimental support for DeepSeek models is available.

*   **LiteLLM (Default)**:  Configure LiteLLM for various LLM providers.

    *   **Option 1: Unified API base for both models**

        *Example: OpenAI*

        ```bash
        cat << EOF  > .env
        CHAT_MODEL=gpt-4o
        EMBEDDING_MODEL=text-embedding-3-small
        OPENAI_API_BASE=<your_unified_api_base>
        OPENAI_API_KEY=<replace_with_your_openai_api_key>
        EOF
        ```

        *Example: Azure OpenAI*

        ```bash
        cat << EOF  > .env
        EMBEDDING_MODEL=azure/<Model deployment supporting embedding>
        CHAT_MODEL=azure/<your deployment name>
        AZURE_API_KEY=<replace_with_your_openai_api_key>
        AZURE_API_BASE=<your_unified_api_base>
        AZURE_API_VERSION=<azure api version>
        EOF
        ```

    *   **Option 2: Separate API bases for Chat and Embedding models**

        ```bash
        cat << EOF  > .env
        CHAT_MODEL=gpt-4o
        OPENAI_API_BASE=<your_chat_api_base>
        OPENAI_API_KEY=<replace_with_your_openai_api_key>

        EMBEDDING_MODEL=litellm_proxy/BAAI/bge-large-en-v1.5
        LITELLM_PROXY_API_KEY=<replace_with_your_siliconflow_api_key>
        LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1
        EOF
        ```

    *   *Example: DeepSeek Setup*

        ```bash
        cat << EOF  > .env
        CHAT_MODEL=deepseek/deepseek-chat
        DEEPSEEK_API_KEY=<replace_with_your_deepseek_api_key>

        EMBEDDING_MODEL=litellm_proxy/BAAI/bge-m3
        LITELLM_PROXY_API_KEY=<replace_with_your_siliconflow_api_key>
        LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1
        EOF
        ```

    *   Set `REASONING_THINK_RM=True` if using reasoning models (e.g., with `<think>` tags).

*   **Validate Configuration:**

    ```bash
    rdagent health_check
    ```

### 🚀 Run the Application

- Run the **Automated Quantitative Trading & Iterative Factors Model Joint Evolution**:  [Qlib](http://github.com/microsoft/qlib) self-loop factor & model proposal and implementation application
  ```sh
  rdagent fin_quant
  ```

- Run the **Automated Quantitative Trading & Iterative Factors Evolution**:  [Qlib](http://github.com/microsoft/qlib) self-loop factor proposal and implementation application
  ```sh
  rdagent fin_factor
  ```

- Run the **Automated Quantitative Trading & Iterative Model Evolution**: [Qlib](http://github.com/microsoft/qlib) self-loop model proposal and implementation application
  ```sh
  rdagent fin_model
  ```

- Run the **Automated Quantitative Trading & Factors Extraction from Financial Reports**:  Run the [Qlib](http://github.com/microsoft/qlib) factor extraction and implementation application based on financial reports
  ```sh
  # 1. Generally, you can run this scenario using the following command:
  rdagent fin_factor_report --report_folder=<Your financial reports folder path>

  # 2. Specifically, you need to prepare some financial reports first. You can follow this concrete example:
  wget https://github.com/SunsetWolf/rdagent_resource/releases/download/reports/all_reports.zip
  unzip all_reports.zip -d git_ignore_folder/reports
  rdagent fin_factor_report --report_folder=git_ignore_folder/reports
  ```

- Run the **Automated Model Research & Development Copilot**: model extraction and implementation application
  ```sh
  # 1. Generally, you can run your own papers/reports with the following command:
  rdagent general_model <Your paper URL>

  # 2. Specifically, you can do it like this. For more details and additional paper examples, use `rdagent general_model -h`:
  rdagent general_model  "https://arxiv.org/pdf/2210.09789"
  ```

- Run the **Automated Medical Prediction Model Evolution**: Medical self-loop model proposal and implementation application

  ```bash
  # Generally, you can run the data science program with the following command:
  rdagent data_science --competition <your competition name>

  # Specifically, you need to create a folder for storing competition files (e.g., competition description file, competition datasets, etc.), and configure the path to the folder in your environment. In addition, you need to use chromedriver when you download the competition descriptors, which you can follow for this specific example:

  # 1. Download the dataset, extract it to the target folder.
  wget https://github.com/SunsetWolf/rdagent_resource/releases/download/ds_data/arf-12-hours-prediction-task.zip
  unzip arf-12-hours-prediction-task.zip -d ./git_ignore_folder/ds_data/

  # 2. Configure environment variables in the `.env` file
  dotenv set DS_LOCAL_DATA_PATH "$(pwd)/git_ignore_folder/ds_data"
  dotenv set DS_CODER_ON_WHOLE_PIPELINE True
  dotenv set DS_IF_USING_MLE_DATA False
  dotenv set DS_SAMPLE_DATA_BY_LLM False
  dotenv set DS_SCEN rdagent.scenarios.data_science.scen.DataScienceScen

  # 3. run the application
  rdagent data_science --competition arf-12-hours-prediction-task
  ```

  **NOTE:** For more information about the dataset, please refer to the [documentation](https://rdagent.readthedocs.io/en/latest/scens/data_science.html).

- Run the **Automated Kaggle Model Tuning & Feature Engineering**:  self-loop model proposal and feature engineering implementation application <br />
  > Using **tabular-playground-series-dec-2021** as an example. <br />
  > 1. Register and login on the [Kaggle](https://www.kaggle.com/) website. <br />
  > 2. Configuring the Kaggle API. <br />
  > (1) Click on the avatar (usually in the top right corner of the page) -> `Settings` -> `Create New Token`, A file called `kaggle.json` will be downloaded. <br />
  > (2) Move `kaggle.json` to `~/.config/kaggle/` <br />
  > (3) Modify the permissions of the kaggle.json file. Reference command: `chmod 600 ~/.config/kaggle/kaggle.json` <br />
  > 3. Join the competition: Click `Join the competition` -> `I Understand and Accept` at the bottom of the [competition details page](https://www.kaggle.com/competitions/tabular-playground-series-dec-2021/data).
  ```bash
  # Generally, you can run the Kaggle competition program with the following command:
  rdagent data_science --competition <your competition name>

  # 1. Configure environment variables in the `.env` file
  mkdir -p ./git_ignore_folder/ds_data
  dotenv set DS_LOCAL_DATA_PATH "$(pwd)/git_ignore_folder/ds_data"
  dotenv set DS_CODER_ON_WHOLE_PIPELINE True
  dotenv set DS_IF_USING_MLE_DATA True
  dotenv set DS_SAMPLE_DATA_BY_LLM True
  dotenv set DS_SCEN rdagent.scenarios.data_science.scen.KaggleScen

  # 2. run the application
  rdagent data_science --competition tabular-playground-series-dec-2021
  ```

### 🖥️ Monitor the Application Results

```bash
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

- About the `data_science` parameter: If you want to see the logs of the data science scenario, set the `data_science` parameter to `True`; otherwise set it to `False`.
 
- Although port 19899 is not commonly used, but before you run this demo, you need to check if port 19899 is occupied. If it is, please change it to another port that is not occupied.

  You can check if a port is occupied by running the following command.

  ```sh
  rdagent health_check --no-check-env --no-check-docker
  ```

---

## 🏭 Scenarios

RD-Agent is designed for multiple data-driven industrial scenarios.

### 🎯 Goal: Agent for Data-driven R&D

The project focuses on building an Agent to automate data-driven R&D by:

*   Reading real-world data (reports, papers) to extract key elements (formulas, features, models).
*   Implementing extracted information into runnable code.
*   Proposing new ideas based on existing knowledge.

<img src="docs/_static/overview.png" alt="Overview of RD-Agent" width="85%">

### 📈 Scenarios

RD-Agent supports two main roles in data-driven scenarios: 🦾Copilot and 🤖Agent.

| Scenario/Target | Model Implementation                   | Data Building                                                                      |
| --------------- | -------------------------------------- | ---------------------------------------------------------------------------------- |
| **💹 Finance**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop)[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s) |  🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s) <br/>   🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)  |
| **🩺 Medical**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm)[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4) | -                                                                                  |
| **🏭 General**      | 🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model)[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKQ7o) <br/> 🤖 Auto Kaggle Model Tuning   | 🤖Auto Kaggle feature Engineering |

*   **[RoadMap](https://rdagent.readthedocs.io/en/latest/scens/data_science.html#roadmap)**: The project roadmap focuses on enhancing the Kaggle scenario.

Refer to the scenario documentation for detailed setup instructions: **[📖readthedocs_scen](https://rdagent.readthedocs.io/en/latest/scens/catalog.html)**

---

## ⚙️ Framework

The RD-Agent Framework:
<div align="center">
  <img src="docs/_static/Framework-RDAgent.png" alt="RD-Agent Framework" width="85%">
</div>

Research questions within this framework are divided into:

| Research Area             | Paper/Work List              |
| ------------------------- | ---------------------------- |
| **Benchmark R&D abilities** | [Benchmark](#benchmark)      |
| **Idea Proposal**         | [Research](#research)        |
| **Idea Realization**      | [Development](#development) |

The project emphasizes continuous improvement and learning.

---

## 📃 Paper/Work List

### **Overall Technical Report**

*   [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)

```bibtex
@misc{yang2024rdagent,
    title={R\&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution},
    author={Xu Yang and Xiao Yang and Shikai Fang and Bowen Xian and Yuante Li and Jian Wang and Minrui Xu and Haoran Pan and Xinpeng Hong and Weiqing Liu and Yelong Shen and Weizhu Chen and Jiang Bian},
    year={2025},
    eprint={2505.14738},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2505.14738}
}
```

### **📊 Benchmark**

*   [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)

```bibtex
@misc{chen2024datacentric,
    title={Towards Data-Centric Automatic R&D},
    author={Haotian Chen and Xinjie Shen and Zeqi Ye and Wenjun Feng and Haoxue Wang and Xiao Yang and Xu Yang and Weiqing Liu and Jiang Bian},
    year={2024},
    eprint={2404.11276},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

### **🔍 Research**

The framework focuses on iteratively proposing, verifying, and getting feedback on real-world applications.

### **🛠️ Development**

*   [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)

```bibtex
@misc{yang2024collaborative,
    title={Collaborative Evolving Strategy for Automatic Data-Centric Development},
    author={Xu Yang and Haotian Chen and Wenjun Feng and Haoxue Wang and Zeqi Ye and Xinjie Shen and Xiao Yang and Shizhao Sun and Weiqing Liu and Jiang Bian},
    year={2024},
    eprint={2407.18690},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

### **Deep Application in Diverse Scenarios**

*   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

```bibtex
@misc{li2025rdagentquant,
    title={R\&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization},
    author={Yuante Li and Xu Yang and Xiao Yang and Minrui Xu and Xisen Wang and Weiqing Liu and Jiang Bian},
    year={2025},
    eprint={2505.15155},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

---

## 🤝 Contributing

Contribute to RD-Agent!  Refer to the [Contributing Guide](CONTRIBUTING.md) for details.

Before submitting a pull request, ensure that your code passes all CI checks.

---

## 📝 Guidelines

*   The project welcomes contributions and suggestions.
*   Explore the issues list or search for `TODO:` in the codebase.

<img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>

<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

*Note: Contributions from key team members prior to the open-source release may not be explicitly listed in the current commit history.*

---

## ⚖️ Legal Disclaimer

*The RD-agent is provided “as is”, without warranty... (See full disclaimer in the original README)*

---