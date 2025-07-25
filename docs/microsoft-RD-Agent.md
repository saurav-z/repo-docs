<h1 align="center">🤖 R&D-Agent: Automate Your Data-Driven Research & Development</h1>

<div align="center">
  <img src="docs/_static/logo.png" alt="R&D-Agent logo" style="width:25%; margin-bottom: 1em;">
</div>

R&D-Agent is a cutting-edge, multi-agent framework designed to **automate and accelerate the entire data-driven R&D process**, from idea generation to implementation and evaluation.  Explore how R&D-Agent leads the way in automated ML Engineering, featuring state-of-the-art performance!

[🖥️ Live Demo](https://rdagent.azurewebsites.net/) |
[🎥 Demo Video](https://rdagent.azurewebsites.net/factor_loop) |
[📖 Documentation](https://rdagent.readthedocs.io/en/latest/index.html) |
[📄 Tech Report](https://aka.ms/RD-Agent-Tech-Report) |
[▶️ YouTube](https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR) |
[📃 Papers](#-paperwork-list)

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
[![Documentation Status](https://readthedocs.org/projects/rdagent/badge/?version=latest)](https://rdagent.readthedocs.io/en/latest/?badge=latest)
[![Readthedocs Preview](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml) <!-- this badge is too long, please place it in the last one to make it pretty --> 
[![arXiv](https://img.shields.io/badge/arXiv-2505.14738-00ff00.svg)](https://arxiv.org/abs/2505.14738)

## 🔑 Key Features

*   **Automated R&D:** Automates critical R&D tasks, significantly boosting productivity.
*   **Multi-Agent Framework:**  Facilitates complex workflows through coordinated agents.
*   **Data-Centric Approach:** Focuses on data-driven solutions, optimizing models and datasets.
*   **Kaggle & Data Science Integration:**  Supports automated model tuning and feature engineering for data science competitions.
*   **Financial Applications:** Automates quant strategy development and research.
*   **Extensible:**  Easily adaptable to a wide range of R&D scenarios.
*   **[LiteLLM](https://github.com/BerriAI/litellm)** Support

## 🏆 Leading Performance on MLE-bench

R&D-Agent currently leads as the top-performing machine learning engineering agent on MLE-bench. Explore the benchmark results:

| Agent | Low == Lite (%) | Medium (%) | High (%) | All (%) |
|---------|--------|-----------|---------|----------|
| R&D-Agent o1-preview | 48.18 ± 2.49 | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21 | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview | 34.3 ± 2.4 | 8.8 ± 1.1 | 10.0 ± 1.9 | 16.9 ± 1.1 |

**Notes:**
-   **O3(R)+GPT-4.1(D)**: This version is designed to both reduce average time per loop and leverage a cost-effective combination of backend LLMs by seamlessly integrating Research Agent (o3) with Development Agent (GPT-4.1).
-   **AIDE o1-preview**: Represents the previously best public result on MLE-bench as reported in the original MLE-bench paper.
-   Average and standard deviation results for R&D-Agent o1-preview is based on a independent of 5 seeds and for R&D-Agent o3(R)+GPT-4.1(D) is based on 6 seeds.
-   According to MLE-Bench, the 75 competitions are categorized into three levels of complexity: **Low==Lite** if we estimate that an experienced ML engineer can produce a sensible solution in under 2 hours, excluding the time taken to train any models; **Medium** if it takes between 2 and 10 hours; and **High** if it takes more than 10 hours.

-   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
-   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

For running R&D-Agent on MLE-bench, refer to **[MLE-bench Guide: Running ML Engineering via MLE-bench](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)**

## 🤖 Scenarios & Demos

R&D-Agent can be applied across a broad range of data-driven scenarios, serving as both a Copilot for automation and an Agent for autonomous idea generation.

**Demo Videos and Live Examples:**

*   **Automated Quant Trading:** 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop)[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s) & 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s).
*   **Automated Finance:**  🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c).
*   **Automated Medical Model Evolution** 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm)[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4)
*   **General R&D:**  🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model)[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKa7o) & 🤖 Auto Kaggle Model Tuning & 🤖 Auto Kaggle feature Engineering.

  Different scenarios vary in entrance and configuration. Please check the detailed setup tutorial in the scenarios documents.

Here is a gallery of [successful explorations](https://github.com/SunsetWolf/rdagent_resource/releases/download/demo_traces/demo_traces.zip) (5 traces showed in **[🖥️ Live Demo](https://rdagent.azurewebsites.net/)**). You can download and view the execution trace using [this command](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#%EF%B8%8F-monitor-the-application-results) from the documentation.

## 🚀 Quick Start

### Prerequisites

*   **Linux:**  R&D-Agent currently supports Linux.
*   **Docker:** Ensure Docker is installed.  Follow instructions [here](https://docs.docker.com/engine/install/).  Verify with `docker run hello-world`.
*   **Conda Environment:**
    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```

### Installation

#### For Users
```bash
pip install rdagent
```

#### For Developers
```bash
git clone https://github.com/microsoft/RD-Agent
cd RD-Agent
make dev
```

More details can be found in the [development setup](https://rdagent.readthedocs.io/en/latest/development.html).

### Health Check

```bash
rdagent health_check --no-check-env
```

### Configuration

Configure your LLM and embedding models in the `.env` file using one of the following options:

> **🔥 Attention**: We now provide experimental support for **DeepSeek** models! You can use DeepSeek's official API for cost-effective and high-performance inference. See the configuration example below for DeepSeek setup.

**Option 1: Unified API base for both models**

```bash
cat << EOF  > .env
# Set to any model supported by LiteLLM.
CHAT_MODEL=gpt-4o 
EMBEDDING_MODEL=text-embedding-3-small
# Configure unified API base
OPENAI_API_BASE=<your_unified_api_base>
OPENAI_API_KEY=<replace_with_your_openai_api_key>
EOF
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
EOF
```

**Configuration Example: DeepSeek Setup**:

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
EOF
```

Notice: If you are using reasoning models that include thought processes in their responses (such as \<think> tags), you need to set the following environment variable:
```bash
REASONING_THINK_RM=True
```

For more detailed configuration, see the [documentation](https://rdagent.readthedocs.io/en/latest/installation_and_configuration.html).

### Verify Configuration
```bash
rdagent health_check
```

### ⚙️ Run Applications

Execute the following commands to run the demos:

*   **Automated Quant Trading & Iterative Factors Model Joint Evolution:**
    ```bash
    rdagent fin_quant
    ```

*   **Automated Quant Trading & Iterative Factors Evolution:**
    ```bash
    rdagent fin_factor
    ```

*   **Automated Quant Trading & Iterative Model Evolution:**
    ```bash
    rdagent fin_model
    ```

*   **Automated Quant Trading & Factors Extraction from Financial Reports:**
    ```bash
    # 1. Generally, you can run this scenario using the following command:
    rdagent fin_factor_report --report_folder=<Your financial reports folder path>

    # 2. Specifically, you need to prepare some financial reports first. You can follow this concrete example:
    wget https://github.com/SunsetWolf/rdagent_resource/releases/download/reports/all_reports.zip
    unzip all_reports.zip -d git_ignore_folder/reports
    rdagent fin_factor_report --report_folder=git_ignore_folder/reports
    ```

*   **Automated Model Research & Development Copilot:**
    ```bash
    # 1. Generally, you can run your own papers/reports with the following command:
    rdagent general_model <Your paper URL>

    # 2. Specifically, you can do it like this. For more details and additional paper examples, use `rdagent general_model -h`:
    rdagent general_model  "https://arxiv.org/pdf/2210.09789"
    ```

*   **Automated Medical Prediction Model Evolution:**
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

*   **Automated Kaggle Model Tuning & Feature Engineering:**
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

### 🖥️ Monitor Results

```bash
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

## 🏭 Scenarios Overview

R&D-Agent is designed for various data-driven R&D applications.

### 🎯 Goal: Agent for Data-driven R&D
*   **Framework:** R&D-Agent streamlines the development of models and data by automating the industrial R&D process.
*   **Methodology:**  It incorporates two primary components: "R" (proposing ideas) and "D" (implementing them).
*   **Benefits:** Automates and evolves R&D, generating solutions with significant industrial value.

### 📈 Scenarios/Demos

| Scenario/Target | Model Implementation                                  | Data Building                                                                      |
| :--------------- | :---------------------------------------------------- | :--------------------------------------------------------------------------------- |
| **💹 Finance**       | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop) | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) <br/>   🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)  |
| **🩺 Medical**       | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm)          | -                                                                                  |
| **🏭 General**       | 🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model) <br/> 🤖 Auto Kaggle Model Tuning  | 🤖Auto Kaggle feature Engineering |

## ⚙️ Framework

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

R&D-Agent offers a comprehensive framework for automating the R&D process in data science. Key research areas include:
*   **Benchmark the R&D abilities**
*   **Idea proposal:** Explore new ideas or refine existing ones
*   **Ability to realize ideas:** Implement and execute ideas

More information can be found at the [official documentation](https://rdagent.readthedocs.io/).

## 📃 Paper/Work List

### Overall Technical Report
*   [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)

### 📊 Benchmark
*   [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)

### 🔍 Research

R&D-Agent has established a scientific research automation framework that supports linking with real-world verification. Explore how you can apply research in the [Live Demo](https://rdagent.azurewebsites.net/).

### 🛠️ Development
*   [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)

### Deep Application in Diverse Scenarios
*   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

## 🤝 Contributing

Contribute to R&D-Agent and help improve it further!  See the [Contributing Guide](CONTRIBUTING.md) for details.  Remember to run CI checks before submitting a pull request.

## ⚖️ Legal Disclaimer

*RD-Agent is provided "as is" without warranty.  It is intended for research and development and is not for financial investment or advice. Users are responsible for assessing risks, ensuring responsible AI use, and complying with all applicable laws.*

[**View the original repository on GitHub**](https://github.com/microsoft/RD-Agent)