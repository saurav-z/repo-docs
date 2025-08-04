<h1 align="center">🤖 R&D-Agent: Automating Data-Driven Innovation</h1>

<div align="center">
  <img src="docs/_static/logo.png" alt="RD-Agent logo" style="width:50%; ">
</div>

**R&D-Agent is a cutting-edge, multi-agent framework designed to revolutionize research and development across various data-driven domains.** Learn how to harness the power of AI agents for automated innovation and efficient ML engineering.

[🖥️ Live Demo](https://rdagent.azurewebsites.net/) | [🎥 Demo Video](https://rdagent.azurewebsites.net/factor_loop) | [▶️ YouTube Channel](https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR) | [📖 Documentation](https://rdagent.readthedocs.io/en/latest/index.html) | [📄 Tech Report](https://aka.ms/RD-Agent-Tech-Report) | [📃 Papers](#-paperwork-list)

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

## Key Features

*   **Automated R&D:** Automates critical industrial R&D processes, focusing on data-driven scenarios.
*   **Multi-Agent Framework:**  Employs coordinated agents for tasks like research (idea generation), development (implementation), and evolution (improvement).
*   **Data-Centric Approach:**  Focuses on automating model and data development.
*   **Real-World Performance:** Demonstrated superior results in benchmarks and real-world financial applications.
*   **Extensible and Customizable:**  Offers flexible configuration options and supports various LLM providers.
*   **Comprehensive Documentation:**  Includes detailed documentation and guides for installation, configuration, and usage.
*   **Community Support:** Active Discord channel for support and discussions.

## The Best Machine Learning Engineering Agent on MLE-bench!

R&D-Agent is the top-performing machine learning engineering agent on the [MLE-bench](https://github.com/openai/mle-bench) benchmark, outperforming other agents in real-world ML engineering scenarios.

| Agent | Low == Lite (%) | Medium (%) | High (%) | All (%) |
|---------|--------|-----------|---------|----------|
| R&D-Agent o1-preview | 48.18 ± 2.49 | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21 | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview | 34.3 ± 2.4 | 8.8 ± 1.1 | 10.0 ± 1.9 | 16.9 ± 1.1 |

See the full results:

*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

For running R&D-Agent on MLE-bench, refer to **[MLE-bench Guide: Running ML Engineering via MLE-bench](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)**

## Use Cases

*   **Automated Quant Factory:** Automate the full-stack research and development of quantitative strategies via coordinated factor-model co-optimization.  [🎥Demo Video](https://rdagent.azurewebsites.net/factor_loop) | [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s)
*   **Data Mining Agent:** Iteratively propose data & models [🎥Demo Video 1](https://rdagent.azurewebsites.net/model_loop) | [▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s) and [🎥Demo Video 2](https://rdagent.azurewebsites.net/dmm) | [▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4), implementing by gaining knowledge from data.
*   **Research Copilot:** Auto read research papers [🎥Demo Video](https://rdagent.azurewebsites.net/report_model) | [▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKQ7o) / financial reports [🎥Demo Video](https://rdagent.azurewebsites.net/report_factor) | [▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c) and implement model structures or building datasets.
*   **Kaggle Agent:** Auto Model Tuning and Feature Engineering [🎥Demo Video Coming Soon...] and implementing them to achieve more in competitions.

## ⚡ Quick Start

**RD-Agent currently only supports Linux.**

### 🐳 Docker Installation
Ensure Docker is installed and the current user can run Docker commands without `sudo`.

### 🐍 Create a Conda Environment
```bash
conda create -n rdagent python=3.10
conda activate rdagent
```

### 🛠️ Install R&D-Agent

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

### 💊 Health Check
```bash
rdagent health_check --no-check-env
```

### ⚙️ Configuration

Configure your environment variables (e.g., API keys, model choices). Example using LiteLLM:

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

**Configuration Example: DeepSeek Setup**:

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
```

Notice: If you are using reasoning models that include thought processes in their responses (such as \<think> tags), you need to set the following environment variable:
```bash
REASONING_THINK_RM=True
```

Refer to the [documentation](https://rdagent.readthedocs.io/en/latest/installation_and_configuration.html) for detailed configuration instructions.

```bash
rdagent health_check
```

### 🚀 Run the Application
Run the following commands for the demos:
* Run the **Automated Quantitative Trading & Iterative Factors Model Joint Evolution**:  [Qlib](http://github.com/microsoft/qlib) self-loop factor & model proposal and implementation application
  ```sh
  rdagent fin_quant
  ```
* Run the **Automated Quantitative Trading & Iterative Factors Evolution**:  [Qlib](http://github.com/microsoft/qlib) self-loop factor proposal and implementation application
  ```sh
  rdagent fin_factor
  ```
* Run the **Automated Quantitative Trading & Iterative Model Evolution**: [Qlib](http://github.com/microsoft/qlib) self-loop model proposal and implementation application
  ```sh
  rdagent fin_model
  ```
* Run the **Automated Quantitative Trading & Factors Extraction from Financial Reports**:  Run the [Qlib](http://github.com/microsoft/qlib) factor extraction and implementation application based on financial reports
  ```sh
  # 1. Generally, you can run this scenario using the following command:
  rdagent fin_factor_report --report_folder=<Your financial reports folder path>

  # 2. Specifically, you need to prepare some financial reports first. You can follow this concrete example:
  wget https://github.com/SunsetWolf/rdagent_resource/releases/download/reports/all_reports.zip
  unzip all_reports.zip -d git_ignore_folder/reports
  rdagent fin_factor_report --report_folder=git_ignore_folder/reports
  ```
* Run the **Automated Model Research & Development Copilot**: model extraction and implementation application
  ```sh
  # 1. Generally, you can run your own papers/reports with the following command:
  rdagent general_model <Your paper URL>

  # 2. Specifically, you can do it like this. For more details and additional paper examples, use `rdagent general_model -h`:
  rdagent general_model  "https://arxiv.org/pdf/2210.09789"
  ```
* Run the **Automated Medical Prediction Model Evolution**: Medical self-loop model proposal and implementation application

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

* Run the **Automated Kaggle Model Tuning & Feature Engineering**:  self-loop model proposal and feature engineering implementation application <br />
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

## 🏭 Scenarios

R&D-Agent supports various data-driven industrial scenarios:

### 🎯 Goal: Agent for Data-driven R&D

R&D-Agent aims to automate Data-Driven R&D by:

*   Reading real-world material (reports, papers, etc.) and **extracting** key formulas, feature descriptions, and model details.
*   **Implementing** these extracted components into executable code.
*   **Proposing new ideas** based on existing knowledge and observations.

### 📈 Supported Scenarios/Demos

| Scenario/Target | Model Implementation                   | Data Building                                                                      |
| --              | --                                     | --                                                                                 |
| **💹 Finance**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop)[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s) |  🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s) <br/>   🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)  |
| **🩺 Medical**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm)[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4) | -                                                                                  |
| **🏭 General**      | 🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model)[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKa7o) <br/> 🤖 Auto Kaggle Model Tuning   | 🤖Auto Kaggle feature Engineering |

*   **[RoadMap](https://rdagent.readthedocs.io/en/latest/scens/data_science.html#roadmap)**: Currently, we are working hard to add new features to the Kaggle scenario.

Refer to **[📖readthedocs_scen](https://rdagent.readthedocs.io/en/latest/scens/catalog.html)** for scenario details.

## ⚙️ Framework

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

The research questions within this framework can be divided into three main categories:
| Research Area | Paper/Work List |
|--------------------|-----------------|
| **Benchmark the R&D abilities** | [Benchmark](#benchmark) |
| **Idea proposal:** Explore new ideas or refine existing ones | [Research](#research) |
| **Ability to realize ideas:** Implement and execute ideas | [Development](#development) |

More documents can be found in the **[📖 readthedocs](https://rdagent.readthedocs.io/)**.

## 📃 Paper/Work List

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

In a data mining expert's daily research and development process, they propose a hypothesis (e.g., a model structure like RNN can capture patterns in time-series data), design experiments (e.g., finance data contains time-series and we can verify the hypothesis in this scenario), implement the experiment as code (e.g., Pytorch model structure), and then execute the code to get feedback (e.g., metrics, loss curve, etc.). The experts learn from the feedback and improve in the next iteration.

Based on the principles above, we have established a basic method framework that continuously proposes hypotheses, verifies them, and gets feedback from the real-world practice. This is the first scientific research automation framework that supports linking with real-world verification.

For more detail, please refer to our **[🖥️ Live Demo page](https://rdagent.azurewebsites.net)**.

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

We welcome contributions! See the [Contributing Guide](CONTRIBUTING.md) for details.

## ⚖️ Legal Disclaimer

```text
The RD-agent is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. The RD-agent is aimed to facilitate research and development process in the financial industry and not ready-to-use for any financial investment or advice. Users shall independently assess and test the risks of the RD-agent in a specific use scenario, ensure the responsible use of AI technology, including but not limited to developing and integrating risk mitigation measures, and comply with all applicable laws and regulations in all applicable jurisdictions. The RD-agent does not provide financial opinions or reflect the opinions of Microsoft, nor is it designed to replace the role of qualified financial professionals in formulating, assessing, and approving finance products. The inputs and outputs of the RD-agent belong to the users and users shall assume all liability under any theory of liability, whether in contract, torts, regulatory, negligence, products liability, or otherwise, associated with use of the RD-agent and any inputs and outputs thereof.
```

[Back to Top](#-rd-agent-automating-data-driven-innovation)