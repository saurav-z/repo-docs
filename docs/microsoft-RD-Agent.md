<h1 align="center">
  <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:70%;">
  RD-Agent: Automating Data-Driven R&D with AI
</h1>

<p align="center">
  <b>Supercharge your research and development with RD-Agent, the leading AI-powered agent for machine learning engineering and quantitative finance.</b>
</p>

<div align="center">
  <a href="https://rdagent.azurewebsites.net" target="_blank">🖥️ Live Demo</a> |
  <a href="https://rdagent.azurewebsites.net/factor_loop" target="_blank">🎥 Demo Video</a>  |
  <a href="https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR" target="_blank">▶️ YouTube Channel</a>  |
  <a href="https://rdagent.readthedocs.io/en/latest/index.html" target="_blank">📖 Documentation</a> |
  <a href="https://aka.ms/RD-Agent-Tech-Report" target="_blank">📄 Tech Report</a> |
  <a href="#-paperwork-list"> 📃 Papers </a> |
  <a href="https://discord.gg/ybQ97B6Jjy">💬 Discord</a>
</div>

<p align="center">
  <a href="https://github.com/microsoft/RD-Agent">
    <img alt="GitHub Workflow Status" src="https://github.com/microsoft/RD-Agent/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://github.com/microsoft/RD-Agent/actions/workflows/github-code-scanning/codeql/badge.svg">
    <img alt="CodeQL" src="https://github.com/microsoft/RD-Agent/actions/workflows/github-code-scanning/codeql/badge.svg">
  </a>
  <a href="https://github.com/microsoft/RD-Agent/actions/workflows/dependabot/dependabot-updates/badge.svg">
    <img alt="Dependabot Updates" src="https://github.com/microsoft/RD-Agent/actions/workflows/dependabot/dependabot-updates/badge.svg">
  </a>
  <a href="https://github.com/microsoft/RD-Agent/actions/workflows/pr.yml/badge.svg">
    <img alt="Lint PR Title" src="https://github.com/microsoft/RD-Agent/actions/workflows/pr.yml/badge.svg">
  </a>
  <a href="https://github.com/microsoft/RD-Agent/actions/workflows/release.yml/badge.svg">
    <img alt="Release" src="https://github.com/microsoft/RD-Agent/actions/workflows/release.yml/badge.svg">
  </a>
  <a href="https://pypi.org/project/rdagent/#files">
    <img alt="Platform" src="https://img.shields.io/badge/platform-Linux-blue">
  </a>
  <a href="https://pypi.org/project/rdagent/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/rdagent">
  </a>
  <a href="https://pypi.org/project/rdagent/">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/rdagent">
  </a>
  <a href="https://github.com/microsoft/RD-Agent/releases">
    <img alt="Release" src="https://img.shields.io/github/v/release/microsoft/RD-Agent">
  </a>
    <a href="https://github.com/microsoft/RD-Agent/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/microsoft/RD-Agent">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit">
  </a>
  <a href="http://mypy-lang.org/">
    <img alt="Checked with mypy" src="https://www.mypy-lang.org/static/mypy_badge.svg">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json">
  </a>
  <a href="https://readthedocs.org/projects/rdagent/badge/?version=latest">
    <img alt="Documentation Status" src="https://readthedocs.org/projects/rdagent/badge/?version=latest">
  </a>
  <a href="https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml/badge.svg">
    <img alt="Readthedocs Preview" src="https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml/badge.svg">
  </a>
  <a href="https://arxiv.org/abs/2505.14738">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.14738-00ff00.svg">
  </a>
</p>

## Key Features

*   **Automated R&D:** RD-Agent automates the most critical and valuable aspects of the R&D process, focusing on data-driven scenarios.
*   **Leading Performance:**  RD-Agent is the top-performing machine learning engineering agent on the [MLE-bench](https://github.com/openai/mle-bench) benchmark.
*   **Data-Centric Quant Framework:** RD-Agent (Q) is the first multi-agent framework designed to automate the full-stack research and development of quantitative strategies via coordinated factor-model co-optimization.
*   **Multi-Scenario Support:** Supports various data-driven scenarios, including automated quantitative trading, data mining, research copilot, and Kaggle competitions.
*   **Modular Design:**  The framework comprises 'R' (proposing ideas) and 'D' (implementing ideas) components for iterative improvement.
*   **Easy to Get Started**: Docker installation and Python environments setup.
*   **LiteLLM Support**: Fully support for LiteLLM as our default backend for integration with multiple LLM providers.

## 🥇 The Best Machine Learning Engineering Agent!

RD-Agent is a leader in the field, demonstrating superior performance on the MLE-bench benchmark, evaluating AI agents on machine learning engineering tasks. The framework achieves strong results on complex tasks.

### MLE-Bench Performance

| Agent | Low == Lite (%) | Medium (%) | High (%) | All (%) |
|---------|--------|-----------|---------|----------|
| R&D-Agent o1-preview | 48.18 ± 2.49 | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21 | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview | 34.3 ± 2.4 | 8.8 ± 1.1 | 10.0 ± 1.9 | 16.9 ± 1.1 |

*   **O3(R)+GPT-4.1(D):** Combines Research Agent (o3) with Development Agent (GPT-4.1) for optimized performance.
*   **AIDE o1-preview:** The previously best public result on MLE-bench.
*   Results are based on independent runs and caterogized to the compettition levels: **Low==Lite** , **Medium** and **High**

Detailed runs:

*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

For running R&D-Agent on MLE-bench, refer to **[MLE-bench Guide: Running ML Engineering via MLE-bench](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)**

## 🥇 The First Data-Centric Quant Multi-Agent Framework!

RD-Agent for Quantitative Finance (RD-Agent(Q)) is the first of its kind, a data-centric, multi-agent framework designed to automate quantitative strategy development.

![RD-Agent(Q) Architecture](https://github.com/user-attachments/assets/3198bc10-47ba-4ee0-8a8e-46d5ce44f45d)

RD-Agent(Q) delivers high ARR (approximately 2x) compared to benchmark factor libraries, using significantly fewer factors and outperforms state-of-the-art time-series models.

Learn more:

*   [RD-Agent(Q) Paper](https://arxiv.org/abs/2505.15155)
*   [RD-Agent(Q) Documentation](https://rdagent.readthedocs.io/en/latest/scens/quant_agent_fin.html)

## 📰 News
| 🗞️ News        | 📝 Description                 |
| --            | ------      |
| [Technical Report Release](#overall-technical-report) | Overall framework description and results on MLE-bench |
| [R&D-Agent-Quant Release](#deep-application-in-diverse-scenarios) | Apply R&D-Agent to quant trading |
| MLE-Bench Results Released | R&D-Agent currently leads as the [top-performing machine learning engineering agent](#-the-best-machine-learning-engineering-agent) on MLE-bench |
| Support LiteLLM Backend | We now fully support **[LiteLLM](https://github.com/BerriAI/litellm)** as our default backend for integration with multiple LLM providers. |
| General Data Science Agent | [Data Science Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html) |
| Kaggle Scenario release | We release **[Kaggle Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)**, try the new features!                  |
| Official WeChat group release  | We created a WeChat group, welcome to join! (🗪[QR Code](https://github.com/microsoft/RD-Agent/issues/880)) |
| Official Discord release  | We launch our first chatting channel in Discord (🗪[![Chat](https://img.shields.io/badge/chat-discord-blue)](https://discord.gg/ybQ97B6Jjy)) |
| First release | **R&D-Agent** is released on GitHub |

## Data Science Agent Preview
<p align="center">
    <a href="https://github.com/microsoft/RD-Agent/blob/main/docs/_static/demo.mp4">
        <img src="docs/_static/demo.png" alt="Watch the demo" width="80%">
    </a>
</p>
Check out our demo video showcasing the current progress of our Data Science Agent under development.

## 🌟 Introduction

<div align="center">
      <img src="docs/_static/scen.png" alt="Our focused scenario" style="width:80%; ">
</div>

RD-Agent focuses on data-driven scenarios to streamline model and data development. The framework employs two key components: 'R' for proposing new ideas and 'D' for implementing them, enabling the automation of R&D processes.

<!-- Tag Cloud -->
The scenarios of R&D-Agent can be your
- 💰 **Automatic Quant Factory** ([🎥Demo Video](https://rdagent.azurewebsites.net/factor_loop)|[▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s))
- 🤖 **Data Mining Agent:** Iteratively proposing data & models ([🎥Demo Video 1](https://rdagent.azurewebsites.net/model_loop)|[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s)) ([🎥Demo Video 2](https://rdagent.azurewebsites.net/dmm)|[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4))  and implementing them by gaining knowledge from data.
- 🦾 **Research Copilot:** Auto read research papers ([🎥Demo Video](https://rdagent.azurewebsites.net/report_model)|[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKa7o)) / financial reports ([🎥Demo Video](https://rdagent.azurewebsites.net/report_factor)|[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)) and implement model structures or building datasets.
- 🤖 **Kaggle Agent:** Auto Model Tuning and Feature Engineering([🎥Demo Video Coming Soon...]()) and implementing them to achieve more in competitions.
- ...

Click the links above to view the demo. More methods and scenarios are continuously added to the project.

Explore our examples in the **[🖥️ Live Demo](https://rdagent.azurewebsites.net/)**.

## ⚡ Quick Start

### Prerequisites

*   **Linux:** RD-Agent currently supports Linux only.
*   **Docker:**  Ensure Docker is installed. Refer to the [official Docker page](https://docs.docker.com/engine/install/) for installation instructions.  Verify the current user can run Docker commands without `sudo`.
*   **Python:** Python 3.10 or 3.11 is recommended.

### 🐳 Docker Installation.

Users must ensure Docker is installed before attempting most scenarios. Please refer to the [official 🐳Docker page](https://docs.docker.com/engine/install/) for installation instructions.
Ensure the current user can run Docker commands **without using sudo**. You can verify this by executing `docker run hello-world`.

### 🐍 Create a Conda Environment

```bash
conda create -n rdagent python=3.10
conda activate rdagent
```

### 🛠️ Install RD-Agent

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

More details in the [development setup](https://rdagent.readthedocs.io/en/latest/development.html).

### 💊 Health Check

```bash
rdagent health_check --no-check-env
```

### ⚙️ Configuration

*   The demos require: `ChatCompletion`, `json_mode`, and `embedding query` abilities.
*   Set your Chat and Embedding Models:

    >   **🔥 Attention**: Experimental support for **DeepSeek** models is provided! Use DeepSeek's official API for cost-effective and high-performance inference. See the configuration example for DeepSeek setup.

*   **Using LiteLLM (Default)**:  We now support LiteLLM as a backend. Configure using the following options:

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

    **Configuration Example: DeepSeek Setup**

    >Since many users encounter configuration errors when setting up DeepSeek. Here's a complete working example for DeepSeek Setup:
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

    **Note:** If you are using reasoning models with thought processes (e.g., \<think> tags), set:

    ```bash
    REASONING_THINK_RM=True
    ```

    Deprecated backend configuration can be found in the [documentation](https://rdagent.readthedocs.io/en/latest/installation_and_configuration.html).

### 🚀 Run the Application

**Health check:**

```bash
rdagent health_check
```

The demos are implemented by the following commands (select your preferred demo):

-   Run the **Automated Quantitative Trading & Iterative Factors Model Joint Evolution**:  [Qlib](http://github.com/microsoft/qlib) self-loop factor & model proposal and implementation application

    ```sh
    rdagent fin_quant
    ```

-   Run the **Automated Quantitative Trading & Iterative Factors Evolution**:  [Qlib](http://github.com/microsoft/qlib) self-loop factor proposal and implementation application

    ```sh
    rdagent fin_factor
    ```

-   Run the **Automated Quantitative Trading & Iterative Model Evolution**: [Qlib](http://github.com/microsoft/qlib) self-loop model proposal and implementation application

    ```sh
    rdagent fin_model
    ```

-   Run the **Automated Quantitative Trading & Factors Extraction from Financial Reports**:  Run the [Qlib](http://github.com/microsoft/qlib) factor extraction and implementation application based on financial reports

    ```sh
    # 1. Generally, you can run this scenario using the following command:
    rdagent fin_factor_report --report_folder=<Your financial reports folder path>

    # 2. Specifically, you need to prepare some financial reports first. You can follow this concrete example:
    wget https://github.com/SunsetWolf/rdagent_resource/releases/download/reports/all_reports.zip
    unzip all_reports.zip -d git_ignore_folder/reports
    rdagent fin_factor_report --report_folder=git_ignore_folder/reports
    ```

-   Run the **Automated Model Research & Development Copilot**: model extraction and implementation application

    ```sh
    # 1. Generally, you can run your own papers/reports with the following command:
    rdagent general_model <Your paper URL>

    # 2. Specifically, you can do it like this. For more details and additional paper examples, use `rdagent general_model -h`:
    rdagent general_model  "https://arxiv.org/pdf/2210.09789"
    ```

-   Run the **Automated Medical Prediction Model Evolution**: Medical self-loop model proposal and implementation application

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

-   Run the **Automated Kaggle Model Tuning & Feature Engineering**:  self-loop model proposal and feature engineering implementation application <br />
    >   Using **tabular-playground-series-dec-2021** as an example. <br />
    >   1.  Register and login on the [Kaggle](https://www.kaggle.com/) website. <br />
    >   2.  Configuring the Kaggle API. <br />
    >       (1) Click on the avatar (usually in the top right corner of the page) -> `Settings` -> `Create New Token`, A file called `kaggle.json` will be downloaded. <br />
    >       (2) Move `kaggle.json` to `~/.config/kaggle/` <br />
    >       (3) Modify the permissions of the kaggle.json file. Reference command: `chmod 600 ~/.config/kaggle/kaggle.json` <br />
    >   3.  Join the competition: Click `Join the competition` -> `I Understand and Accept` at the bottom of the [competition details page](https://www.kaggle.com/competitions/tabular-playground-series-dec-2021/data).
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

*   `data_science` parameter: set to `True` to see data science scenario logs, otherwise `False`.
*   Check port 19899 before running.  If occupied, change the port.

## 🏭 Scenarios

RD-Agent provides support for various data-driven industrial scenarios:

## 🎯 Goal: Agent for Data-driven R&D

The goal is to build an agent to automate Data-Driven R&D.

*   Read real-world material and extract key formulas, features, and models.
*   Implement the extracted formulas in runnable code.
*   Propose new ideas based on current knowledge.

## 📈 Scenarios/Demos

Our system acts as a 🦾Copilot and 🤖Agent.

| Scenario/Target | Model Implementation                   | Data Building                                                                      |
| --              | --                                     | --                                                                                 |
| **💹 Finance**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop)[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s) |  🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s) <br/>   🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)  |
| **🩺 Medical**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm)[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4) | -                                                                                  |
| **🏭 General**      | 🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model)[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKa7o) <br/> 🤖 Auto Kaggle Model Tuning   | 🤖Auto Kaggle feature Engineering |

-   **[RoadMap](https://rdagent.readthedocs.io/en/latest/scens/data_science.html#roadmap)**: Kaggle scenario is currently being expanded.

Detailed setup tutorials are available in the scenario documents.

See example traces: [successful explorations](https://github.com/SunsetWolf/rdagent_resource/releases/download/demo_traces/demo_traces.zip).

Refer to **[📖readthedocs_scen](https://rdagent.readthedocs.io/en/latest/scens/catalog.html)** for details.

## ⚙️ Framework

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

We are working on a framework to improve R&D in data science.

Research areas:

| Research Area | Paper/Work List |
|--------------------|-----------------|
| **Benchmark the R&D abilities** | [Benchmark](#benchmark) |
| **Idea proposal:** Explore new ideas or refine existing ones | [Research](#research) |
| **Ability to realize ideas:** Implement and execute ideas | [Development](#development) |

Key to success: Continuous improvement of R&D capabilities.

More in the **[📖 readthedocs](https://rdagent.readthedocs.io/)**.

## 📃 Paper/Work list

### Overall Technical Report

*   [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)

### 📊 Benchmark

*   [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)

### 🔍 Research

We continuously propose and verify hypotheses to get real-world feedback.

### 🛠️ Development

*   [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)

### Deep Application in Diverse Scenarios

*   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

## 🤝 Contributing

We welcome contributions. See the [Contributing Guide](CONTRIBUTING.md).

Before submitting a pull request, ensure your code passes CI checks.

## 📝 Guidelines

This project welcomes contributions. Solving issues, fixing bugs, improving documentation, or correcting typos are valuable contributions.

Explore the issues list or search for `TODO:` in the codebase.

<img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>

<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

**Note:** The internal commit history before release was not preserved.  Contributions from group members, including Haotian Chen, Wenjun Feng, Haoxue Wang, Zeqi Ye, Xinjie Shen, and Jinhui Li, were not included in the public commits.

## ⚖️ Legal disclaimer

<p style="line-height: 1; font-style: italic;">The RD-agent is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. The RD-agent is aimed to facilitate research and development process in the financial industry and not ready-to-use for any financial investment or advice. Users shall independently assess and test the risks of the RD-agent in a specific use scenario, ensure the responsible use of AI technology, including but not limited to developing and integrating risk mitigation measures, and comply with all applicable laws and regulations in all applicable jurisdictions. The RD-agent does not provide financial opinions or reflect the opinions of Microsoft, nor is it designed to replace the role of qualified financial professionals in formulating, assessing, and approving finance products. The inputs and outputs of the RD-agent belong to the users and users shall assume all liability under any theory of liability, whether in contract, torts, regulatory, negligence, products liability, or otherwise, associated with use of the RD-agent and any inputs and outputs thereof.</p>