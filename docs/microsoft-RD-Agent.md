<h1 align="center">RD-Agent: Automate Your Data-Driven R&D with AI</h1>

**Tired of manual data science tasks? RD-Agent is your AI-powered co-pilot, automating research, development, and evolution in data-driven projects.**  [Explore the RD-Agent Repository](https://github.com/microsoft/RD-Agent)

<div align="center">
  <a href="https://rdagent.azurewebsites.net" target="_blank">🖥️ Live Demo</a> |
  <a href="https://rdagent.azurewebsites.net/factor_loop" target="_blank">🎥 Demo Video</a> |
  <a href="https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR" target="_blank">▶️ YouTube Channel</a> |
  <a href="https://rdagent.readthedocs.io/en/latest/index.html" target="_blank">📖 Documentation</a> |
  <a href="https://aka.ms/RD-Agent-Tech-Report" target="_blank">📄 Tech Report</a> |
  <a href="#paper-work-list"> 📃 Papers </a>
</div>

---

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

## Key Features of RD-Agent

*   **Automated Machine Learning Engineering:** Automates critical aspects of the R&D process.
*   **Multi-Agent Framework:** Leverages a multi-agent system to handle complex tasks.
*   **Data-Centric Approach:** Focuses on data-driven scenarios for streamlined development.
*   **LLM Integration:** Powered by Large Language Models for idea generation, code implementation, and more.
*   **Modular Design:**  Offers a flexible framework for various data science and R&D applications.
*   **Quant Finance Focus (RD-Agent(Q)):** Automates the full-stack research and development of quantitative strategies via coordinated factor-model co-optimization.

## RD-Agent in Action: Top Performance in MLE-Bench

RD-Agent is a leading performer on the MLE-bench, a comprehensive benchmark for evaluating AI agents on machine learning engineering tasks.

| Agent | Low == Lite (%) | Medium (%) | High (%) | All (%) |
|---------|--------|-----------|---------|----------|
| R&D-Agent o1-preview | 48.18 ± 2.49 | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21 | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview | 34.3 ± 2.4 | 8.8 ± 1.1 | 10.0 ± 1.9 | 16.9 ± 1.1 |

**Detailed Results:**

*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

Learn more about running RD-Agent on MLE-bench: [MLE-bench Guide](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)

## RD-Agent(Q): Revolutionizing Quantitative Finance

RD-Agent for Quantitative Finance (RD-Agent(Q)) is a groundbreaking, data-centric, multi-agent framework designed to automate the entire R&D cycle of quantitative strategies.

![RD-Agent(Q) Architecture](https://github.com/user-attachments/assets/3198bc10-47ba-4ee0-8a8e-46d5ce44f45d)

**Key Achievements:**

*   Achieves approximately **2x higher ARR** than benchmark factor libraries.
*   Uses **70% fewer factors**.
*   Outperforms state-of-the-art deep time-series models with smaller resource budgets.

**Learn More:**

*   [RD-Agent(Q) Paper](https://arxiv.org/abs/2505.15155)
*   [RD-Agent(Q) Documentation](https://rdagent.readthedocs.io/en/latest/scens/quant_agent_fin.html)

## News & Updates

*   **[Technical Report Release](#overall-technical-report):** Overview of the framework and MLE-bench results.
*   **[R&D-Agent-Quant Release](#deep-application-in-diverse-scenarios):** Application of R&D-Agent to quantitative trading.
*   **MLE-Bench Results:** RD-Agent is the top-performing machine learning engineering agent on MLE-bench.
*   **LiteLLM Support:** Full support for [LiteLLM](https://github.com/BerriAI/litellm) as the default LLM backend.
*   **Data Science Agent:** [Data Science Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)
*   **Kaggle Scenario Release:** [Kaggle Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)
*   **Community:** Join our [Discord](https://discord.gg/ybQ97B6Jjy) and [WeChat](https://github.com/microsoft/RD-Agent/issues/880) groups.

## Data Science Agent Preview

Watch our demo to see the Data Science Agent in action:

https://github.com/user-attachments/assets/3eccbecb-34a4-4c81-bce4-d3f8862f7305

## Getting Started with RD-Agent

### ⚡ Quick Start

RD-Agent currently supports Linux.

#### 🐳 Docker Installation (Recommended)
Ensure Docker is installed. Refer to the [official Docker documentation](https://docs.docker.com/engine/install/) for installation instructions. Verify Docker runs without `sudo` using `docker run hello-world`.

#### 🐍 Conda Environment
1.  Create a Conda environment (Python 3.10 or 3.11 recommended):
    ```bash
    conda create -n rdagent python=3.10
    ```
2.  Activate the environment:
    ```bash
    conda activate rdagent
    ```

#### 🛠️ Install RD-Agent

*   **For Users:**
    ```bash
    pip install rdagent
    ```

*   **For Developers:**
    ```bash
    git clone https://github.com/microsoft/RD-Agent
    cd RD-Agent
    make dev
    ```

#### 💊 Health Check
Ensure your setup is correct with the health check:

```bash
rdagent health_check --no-check-env
```

#### ⚙️ Configuration
*   **Required Capabilities:** ChatCompletion, json_mode, embedding query
*   **LLM Setup:**  Configure your Chat and Embedding models using the .env file.

    *   **Option 1: Using LiteLLM (Default)**
        *   **Unified API base:**
            ```bash
            cat << EOF  > .env
            CHAT_MODEL=gpt-4o
            EMBEDDING_MODEL=text-embedding-3-small
            OPENAI_API_BASE=<your_unified_api_base>
            OPENAI_API_KEY=<replace_with_your_openai_api_key>
            EOF
            ```
        *   **Azure OpenAI Setup:**
             ```bash
            cat << EOF  > .env
            EMBEDDING_MODEL=azure/<Model deployment supporting embedding>
            CHAT_MODEL=azure/<your deployment name>
            AZURE_API_KEY=<replace_with_your_openai_api_key>
            AZURE_API_BASE=<your_unified_api_base>
            AZURE_API_VERSION=<azure api version>
            EOF
            ```
        *   **Separate API bases:**
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
        *   **DeepSeek Setup:**
            ```bash
            cat << EOF  > .env
            CHAT_MODEL=deepseek/deepseek-chat
            DEEPSEEK_API_KEY=<replace_with_your_deepseek_api_key>
            EMBEDDING_MODEL=litellm_proxy/BAAI/bge-m3
            LITELLM_PROXY_API_KEY=<replace_with_your_siliconflow_api_key>
            LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1
            EOF
            ```
        *   **REASONING_THINK_RM:** Set this environment variable if using models with thought processes:
            ```bash
            REASONING_THINK_RM=True
            ```

*   **Verify Configuration:**
    ```bash
    rdagent health_check
    ```

#### 🚀 Run the Application

*   **Automated Quantitative Trading:**
    ```bash
    rdagent fin_quant
    ```
    ```bash
    rdagent fin_factor
    ```
    ```bash
    rdagent fin_model
    ```
*   **Automated Quantitative Trading & Factors Extraction from Financial Reports:**
    ```bash
    rdagent fin_factor_report --report_folder=<Your financial reports folder path>
    ```
*   **Automated Model Research & Development Copilot:**
    ```bash
    rdagent general_model <Your paper URL>
    ```
    ```bash
    rdagent general_model  "https://arxiv.org/pdf/2210.09789"
    ```
*   **Automated Medical Prediction Model Evolution:**
    ```bash
    rdagent data_science --competition arf-12-hours-prediction-task
    ```
*   **Automated Kaggle Model Tuning & Feature Engineering:**
    ```bash
    rdagent data_science --competition tabular-playground-series-dec-2021
    ```
### 🖥️ Monitor the Application Results
* Run the UI:
    ```bash
    rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
    ```

## 🏭 Scenarios

RD-Agent is applied across various data-driven scenarios.

## 🎯 Goal: Agent for Data-driven R&D

The project aims to automate Data-Driven R&D by:
*   📄 Extracting key information from real-world sources (reports, papers).
*   🛠️ Implementing extracted information into runnable code.
*   💡 Proposing new ideas based on knowledge.

| Scenario/Target | Model Implementation                   | Data Building                                                                      |
| --              | --                                     | --                                                                                 |
| **💹 Finance**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop)[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s) |  🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s) <br/>   🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)  |
| **🩺 Medical**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm)[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4) | -                                                                                  |
| **🏭 General**      | 🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model)[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKQ7o) <br/> 🤖 Auto Kaggle Model Tuning   | 🤖Auto Kaggle feature Engineering |

- **[RoadMap](https://rdagent.readthedocs.io/en/latest/scens/data_science.html#roadmap)**: Currently, we are working hard to add new features to the Kaggle scenario.

Please refer to **[📖readthedocs_scen](https://rdagent.readthedocs.io/en/latest/scens/catalog.html)** for more details of the scenarios.

## ⚙️ Framework

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

The RD-Agent framework aims to automate the R&D process, focusing on:

*   **Benchmark the R&D abilities:** [Benchmark](#benchmark)
*   **Idea proposal:** Explore new ideas or refine existing ones | [Research](#research)
*   **Ability to realize ideas:** Implement and execute ideas | [Development](#development)

## 📃 Paper/Work list

### Overall Technical Report
- [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)
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
![image](https://github.com/user-attachments/assets/28b0488d-a546-4fef-8dc5-563ed64a9b4d)

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
![image](https://github.com/user-attachments/assets/494f55d3-de9e-4e73-ba3d-a787e8f9e841)

### 🔍 Research

Our framework enables continuous hypothesis generation, verification, and feedback for real-world validation.

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
![image](https://github.com/user-attachments/assets/75d9769b-0edd-4caf-9d45-57d1e577054b)

### Deep Application in Diverse Scenarios

- [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)
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
![image](https://github.com/user-attachments/assets/3186f67a-c2f8-4b6b-8bb9-a9b959c13866)


## 🤝 Contributing

We welcome contributions!  Review the [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 Guidelines
This project welcomes contributions and suggestions.
Contributing to this project is straightforward and rewarding. Whether it's solving an issue, addressing a bug, enhancing documentation, or even correcting a typo, every contribution is valuable and helps improve R&D-Agent.

To get started, you can explore the issues list, or search for `TODO:` comments in the codebase by running the command `grep -r "TODO:"`.

<img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>

<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

Before we released R&D-Agent as an open-source project on GitHub, it was an internal project within our group. Unfortunately, the internal commit history was not preserved when we removed some confidential code. As a result, some contributions from our group members, including Haotian Chen, Wenjun Feng, Haoxue Wang, Zeqi Ye, Xinjie Shen, and Jinhui Li, were not included in the public commits.

## ⚖️ Legal disclaimer

<p style="line-height: 1; font-style: italic;">The RD-agent is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. The RD-agent is aimed to facilitate research and development process in the financial industry and not ready-to-use for any financial investment or advice. Users shall independently assess and test the risks of the RD-agent in a specific use scenario, ensure the responsible use of AI technology, including but not limited to developing and integrating risk mitigation measures, and comply with all applicable laws and regulations in all applicable jurisdictions. The RD-agent does not provide financial opinions or reflect the opinions of Microsoft, nor is it designed to replace the role of qualified financial professionals in formulating, assessing, and approving finance products. The inputs and outputs of the RD-agent belong to the users and users shall assume all liability under any theory of liability, whether in contract, torts, regulatory, negligence, products liability, or otherwise, associated with use of the RD-agent and any inputs and outputs thereof.</p>