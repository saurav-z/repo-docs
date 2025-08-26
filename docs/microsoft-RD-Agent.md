<h1 align="center">🚀 RD-Agent: Automating Data-Driven R&D with AI</h1>

<p align="center">
    RD-Agent is a cutting-edge, AI-powered framework designed to automate the entire research and development lifecycle, leading to faster innovation and improved results.
    <br>
    <a href="https://github.com/microsoft/RD-Agent"> 
        <img src="https://img.shields.io/github/stars/microsoft/RD-Agent?style=social" alt="Stars">
    </a>
</p>

<p align="center">
  <a href="https://rdagent.azurewebsites.net" target="_blank">🖥️ Live Demo</a> |
  <a href="https://rdagent.azurewebsites.net/factor_loop" target="_blank">🎥 Demo Video</a> <a href="https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR" target="_blank">▶️YouTube</a>   |
  <a href="https://rdagent.readthedocs.io/en/latest/index.html" target="_blank">📖 Documentation</a> |
  <a href="https://aka.ms/RD-Agent-Tech-Report" target="_blank">📄 Tech Report</a> |
  <a href="#-paperwork-list"> 📃 Papers </a>
</p>

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
[![Readthedocs Preview](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml) <!-- this badge is too long, please place it in the last one to make it pretty --> 
[![arXiv](https://img.shields.io/badge/arXiv-2505.14738-00ff00.svg)](https://arxiv.org/abs/2505.14738)

## Key Features

*   **Automated R&D:** Streamline the entire R&D process, from idea generation to implementation and iteration.
*   **Data-Driven Focus:** Designed for data-centric scenarios, empowering you to build better models and datasets.
*   **Multi-Agent Framework:**  Leverages multiple agents to perform tasks in parallel.
*   **Extensible & Customizable:** Easily adapt to diverse scenarios, from finance to medical research, and more.
*   **Leader on MLE-bench:** RD-Agent o1-preview leads on the MLE-bench.

## Top-Performing Machine Learning Engineering Agent on MLE-bench

RD-Agent is at the forefront of automated machine learning, as shown by its top performance on the [MLE-bench](https://github.com/openai/mle-bench).

| Agent | Low == Lite (%) | Medium (%) | High (%) | All (%) |
|---------|--------|-----------|---------|----------|
| R&D-Agent o1-preview | 48.18 ± 2.49 | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21 | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview | 34.3 ± 2.4 | 8.8 ± 1.1 | 10.0 ± 1.9 | 16.9 ± 1.1 |

**Notes:**
- **O3(R)+GPT-4.1(D)**: This version is designed to both reduce average time per loop and leverage a cost-effective combination of backend LLMs by seamlessly integrating Research Agent (o3) with Development Agent (GPT-4.1).
- **AIDE o1-preview**: Represents the previously best public result on MLE-bench as reported in the original MLE-bench paper.
- Average and standard deviation results for R&D-Agent o1-preview is based on a independent of 5 seeds and for R&D-Agent o3(R)+GPT-4.1(D) is based on 6 seeds.
- According to MLE-Bench, the 75 competitions are categorized into three levels of complexity: **Low==Lite** if we estimate that an experienced ML engineer can produce a sensible solution in under 2 hours, excluding the time taken to train any models; **Medium** if it takes between 2 and 10 hours; and **High** if it takes more than 10 hours.

Detailed runs can be found here:
- [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
- [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

For running R&D-Agent on MLE-bench, refer to **[MLE-bench Guide: Running ML Engineering via MLE-bench](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)**

##  RD-Agent(Q): The First Data-Centric Quant Multi-Agent Framework

RD-Agent for Quantitative Finance, or **RD-Agent(Q)**, is the first data-centric, multi-agent framework designed to automate the full-stack research and development of quantitative strategies via coordinated factor-model co-optimization.

![image](https://github.com/user-attachments/assets/3198bc10-47ba-4ee0-8a8e-46d5ce44f45d)

RD-Agent(Q) achieves approximately 2× higher ARR than benchmark factor libraries while using over 70% fewer factors and outperforms state-of-the-art deep time-series models under smaller resource budgets.

Learn more through the [paper](https://arxiv.org/abs/2505.15155) and [documentation](https://rdagent.readthedocs.io/en/latest/scens/quant_agent_fin.html).

## What's New

*   **[Overall Technical Report Release](#overall-technical-report):** Comprehensive framework description and MLE-bench results.
*   **[R&D-Agent-Quant Release](#deep-application-in-diverse-scenarios):** Application of R&D-Agent to quantitative trading.
*   **MLE-Bench Results Released:** R&D-Agent leading the way on MLE-bench!
*   **LiteLLM Backend Support:** Seamlessly integrate with various LLM providers.
*   **New Scenarios:**  Data Science Agent, Kaggle Agent, and more.
*   **Community:** Join our [Discord](https://discord.gg/ybQ97B6Jjy) or [WeChat group](https://github.com/microsoft/RD-Agent/issues/880) to connect with other users.

## Data Science Agent Preview

Check out our demo video showcasing the current progress of our Data Science Agent under development:

https://github.com/user-attachments/assets/3eccbecb-34a4-4c81-bce4-d3f8862f7305

## Introduction

R&D-Agent focuses on automating critical, high-value aspects of the industrial R&D process, starting with data-driven scenarios.
<div align="center">
      <img src="docs/_static/scen.png" alt="Our focused scenario" style="width:80%; ">
</div>
Methodologically, we use a framework with two core components: 'R' for proposing ideas and 'D' for implementing them.

## Use Cases

R&D-Agent can be your:

*   💰 **Automatic Quant Factory**: ([🎥Demo Video](https://rdagent.azurewebsites.net/factor_loop)|[▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s))
*   🤖 **Data Mining Agent**: Iteratively proposing data & models ([🎥Demo Video 1](https://rdagent.azurewebsites.net/model_loop)|[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s)) ([🎥Demo Video 2](https://rdagent.azurewebsites.net/dmm)|[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4))
*   🦾 **Research Copilot**: Auto read research papers ([🎥Demo Video](https://rdagent.azurewebsites.net/report_model)|[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKQ7o)) / financial reports ([🎥Demo Video](https://rdagent.azurewebsites.net/report_factor)|[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c))
*   🤖 **Kaggle Agent**: Auto Model Tuning and Feature Engineering ([🎥Demo Video Coming Soon...]())
*   ...

Explore our **[🖥️ Live Demo](https://rdagent.azurewebsites.net/)** for more examples!

<div align="center">
    <a href="https://rdagent.azurewebsites.net/" target="_blank">
        <img src="docs/_static/demo.png" alt="Watch the demo" width="80%">
    </a>
</div>

## Quick Start

**Prerequisites:**

*   RD-Agent currently only supports Linux.
*   Docker installed (see [Docker Installation Instructions](https://docs.docker.com/engine/install/)).
*   Python 3.10 or 3.11.

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/microsoft/RD-Agent
    cd RD-Agent
    ```

2.  **Create a Conda Environment:**

    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```

3.  **Install RD-Agent:**

    *   **For Users:**

        ```bash
        pip install rdagent
        ```

    *   **For Developers:**

        ```bash
        make dev
        ```
        (See the [development setup](https://rdagent.readthedocs.io/en/latest/development.html) for details.)

### Health Check

```bash
rdagent health_check --no-check-env
```

### Configuration

RD-Agent requires:

*   `ChatCompletion`
*   `json_mode`
*   `embedding query`

Set your Chat and Embedding Models using environment variables.

*   **Using LiteLLM (Default):**

    **Option 1: Unified API base for both models**

    *Configuration Example: `OpenAI` Setup :*

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

    *Configuration Example: `Azure OpenAI` Setup :*

    > Before using this configuration, please confirm in advance that your `Azure OpenAI API key` supports `embedded models`.

    ```bash
    cat << EOF  > .env
    EMBEDDING_MODEL=azure/<Model deployment supporting embedding>
    CHAT_MODEL=azure/<your deployment name>
    AZURE_API_KEY=<replace_with_your_openai_api_key>
    AZURE_API_BASE=<your_unified_api_base>
    AZURE_API_VERSION=<azure api version>
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

    *Configuration Example: `DeepSeek` Setup :*

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

    Notice: If you are using reasoning models that include thought processes in their responses (such as \<think> tags), you need to set the following environment variable:
    ```bash
    REASONING_THINK_RM=True
    ```

    For deprecated backends and more config info, please refer to the [documentation](https://rdagent.readthedocs.io/en/latest/installation_and_configuration.html).

### Verify Configuration

```bash
rdagent health_check
```

### Run the Application

*   **Automated Quantitative Trading & Iterative Factors Model Joint Evolution (Qlib):**

    ```bash
    rdagent fin_quant
    ```

*   **Automated Quantitative Trading & Iterative Factors Evolution (Qlib):**

    ```bash
    rdagent fin_factor
    ```

*   **Automated Quantitative Trading & Iterative Model Evolution (Qlib):**

    ```bash
    rdagent fin_model
    ```

*   **Automated Quantitative Trading & Factors Extraction from Financial Reports:**

    ```bash
    rdagent fin_factor_report --report-folder=<Your financial reports folder path>
    ```

    or, for a specific example:

    ```bash
    wget https://github.com/SunsetWolf/rdagent_resource/releases/download/reports/all_reports.zip
    unzip all_reports.zip -d git_ignore_folder/reports
    rdagent fin_factor_report --report-folder=git_ignore_folder/reports
    ```

*   **Automated Model Research & Development Copilot:**

    ```bash
    rdagent general_model <Your paper URL>
    ```

    or, for a specific example:

    ```bash
    rdagent general_model  "https://arxiv.org/pdf/2210.09789"
    ```

*   **Automated Medical Prediction Model Evolution:**

    ```bash
    rdagent data_science --competition <your competition name>
    ```

    For specific details and dataset setup, consult the [documentation](https://rdagent.readthedocs.io/en/latest/scens/data_science.html).

*   **Automated Kaggle Model Tuning & Feature Engineering:**

    ```bash
    rdagent data_science --competition tabular-playground-series-dec-2021
    ```
    Ensure you've registered on Kaggle, configured the Kaggle API (`kaggle.json`), and accepted the competition rules. See [the Quickstart Guide](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#%EF%B8%8F-quick-start) for detailed steps.

### Monitor Application Results

```bash
rdagent ui --port 19899 --log-dir <your log folder like "log/"> --data_science <True or False>
```

*   `data_science`: Set to `True` to view logs for data science scenarios, `False` otherwise.
*   Check port 19899 is unoccupied before running.

## Scenarios

We have applied R&D-Agent in several valuable data-driven industrial scenarios.

### 🎯 Goal: Agent for Data-driven R&D

This project aims to build an agent that automates Data-Driven R&D by:

*   📄 Reading real-world material (reports, papers, etc.) and **extracting** key formulas, descriptions of interested **features** and **models**.
*   🛠️ **Implementing** these formulas into runnable code with an evolving process.
*   💡 Proposing **new ideas** based on existing knowledge.

### 📈 Scenarios/Demos

| Scenario/Target | Model Implementation                   | Data Building                                                                      |
| --              | --                                     | --                                                                                 |
| **💹 Finance**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop)[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s) |  🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s) <br/>   🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)  |
| **🩺 Medical**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm)[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4) | -                                                                                  |
| **🏭 General**      | 🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model)[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKQ7o) <br/> 🤖 Auto Kaggle Model Tuning   | 🤖Auto Kaggle feature Engineering |

*   **[RoadMap](https://rdagent.readthedocs.io/en/latest/scens/data_science.html#roadmap)**:  We are working on more features for the Kaggle scenario!

Different scenarios vary in entrance and configuration. Please check the detailed setup tutorial in the scenarios documents.

## Framework

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

The research questions within this framework can be divided into three main categories:

| Research Area | Paper/Work List |
|--------------------|-----------------|
| **Benchmark the R&D abilities** | [Benchmark](#benchmark) |
| **Idea proposal:** Explore new ideas or refine existing ones | [Research](#research) |
| **Ability to realize ideas:** Implement and execute ideas | [Development](#development) |

## 📃 Paper/Work List

### Overall Technical Report

-   [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)
```BibTeX
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

### 📊 Benchmark
- [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)
```BibTeX
@misc{chen2024datacentric,
    title={Towards Data-Centric Automatic R\&D},
    author={Haotian Chen and Xinjie Shen and Zeqi Ye and Wenjun Feng and Haoxue Wang and Xiao Yang and Xu Yang and Weiqing Liu and Jiang Bian},
    year={2024},
    eprint={2404.11276},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```
### 🔍 Research

### 🛠️ Development
- [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)
```BibTeX
@misc{yang2024collaborative,
    title={Collaborative Evolving Strategy for Automatic Data-Centric Development},
    author={Xu Yang and Haotian Chen and Wenjun Feng and Haoxue Wang and Zeqi Ye and Xinjie Shen and Xiao Yang and Shizhao Sun and Weiqing Liu and Jiang Bian},
    year={2024},
    eprint={2407.18690},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

### Deep Application in Diverse Scenarios
-   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)
```BibTeX
@misc{li2025rdagentquant,
    title={R\&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization},
    author={Yuante Li and Xu Yang and Xiao Yang and Minrui Xu and Xisen Wang and Weiqing Liu and Jiang Bian},
    year={2025},
    eprint={2505.15155},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

## Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) for more details.

*   Before submitting a pull request, ensure that your code passes the automatic CI checks.

## 📝 Guidelines

This project encourages contributions. Contribute by solving issues, enhancing documentation, or correcting typos. Your input is valuable and helps improve RD-Agent.

Explore the issues list or search the codebase for `TODO:` comments by running `grep -r "TODO:"`.

<img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>

<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

*Before the public release, the internal commit history wasn't preserved.*

## ⚖️ Legal disclaimer

<p style="line-height: 1; font-style: italic;">The RD-agent is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. The RD-agent is aimed to facilitate research and development process in the financial industry and not ready-to-use for any financial investment or advice. Users shall independently assess and test the risks of the RD-agent in a specific use scenario, ensure the responsible use of AI technology, including but not limited to developing and integrating risk mitigation measures, and comply with all applicable laws and regulations in all applicable jurisdictions. The RD-agent does not provide financial opinions or reflect the opinions of Microsoft, nor is it designed to replace the role of qualified financial professionals in formulating, assessing, and approving finance products. The inputs and outputs of the RD-agent belong to the users and users shall assume all liability under any theory of liability, whether in contract, torts, regulatory, negligence, products liability, or otherwise, associated with use of the RD-agent and any inputs and outputs thereof.</p>