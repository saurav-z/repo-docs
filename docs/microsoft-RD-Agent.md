<h1 align="center">🤖 R&D-Agent: Automating Data-Driven Innovation with AI</h1>

<div align="center">
    <img src="docs/_static/logo.png" alt="R&D-Agent Logo" style="width:50%; margin-bottom: 10px;">
    <p><i>Revolutionize your R&D process with R&D-Agent, the leading AI-powered agent for automating data-driven innovation and machine learning engineering.</i></p>
    <a href="https://github.com/microsoft/RD-Agent"> 🚀 View the Repository on GitHub</a>
</div>

---

## Key Features

*   **Automated R&D:** Streamline your research and development workflows, from idea generation to implementation.
*   **Multi-Agent Framework:** Leverage a coordinated system of agents for comprehensive task management.
*   **Data-Centric Design:** Focus on data-driven scenarios for efficient model and data development.
*   **Integration with MLE-bench:**  R&D-Agent currently leads as the top-performing machine learning engineering agent on MLE-bench.
*   **Extensible Scenarios:** Currently supports scenarios in Finance, Medical, and General Data Science.

---

## Highlights

### 🏆 The Best Machine Learning Engineering Agent on MLE-Bench!

R&D-Agent leads in performance on the [MLE-bench](https://github.com/openai/mle-bench) benchmark:

| Agent                      | Low == Lite (%) | Medium (%) | High (%) | All (%) |
| -------------------------- | -------------- | ---------- | -------- | ------- |
| R&D-Agent o1-preview       | 48.18 ± 2.49   | 8.95 ± 2.36  | 18.67 ± 2.98  | 22.4 ± 1.1  |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21   | 7.89 ± 3.33  | 16.67 ± 3.65  | 22.45 ± 2.45 |
| AIDE o1-preview            | 34.3 ± 2.4     | 8.8 ± 1.1  | 10.0 ± 1.9  | 16.9 ± 1.1  |

You can inspect the detailed runs of the above results online:
*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

### 🥇 The First Data-Centric Quant Multi-Agent Framework!

R&D-Agent for Quantitative Finance, in short **RD-Agent(Q)**, is the first data-centric, multi-agent framework designed to automate the full-stack research and development of quantitative strategies via coordinated factor-model co-optimization.

![image](https://github.com/user-attachments/assets/3198bc10-47ba-4ee0-8a8e-46d5ce44f45d)

Extensive experiments in real stock markets show that, at a cost under $10, RD-Agent(Q) achieves approximately 2× higher ARR than benchmark factor libraries while using over 70% fewer factors. It also surpasses state-of-the-art deep time-series models under smaller resource budgets. Its alternating factor–model optimization further delivers excellent trade-off between predictive accuracy and strategy robustness.

You can learn more details about **RD-Agent(Q)** through the [paper](https://arxiv.org/abs/2505.15155) and reproduce it through the [documentation](https://rdagent.readthedocs.io/en/latest/scens/quant_agent_fin.html).

### 📰 News

| 🗞️ News                            | 📝 Description                                                                                                                                                                                              |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Technical Report Release](#overall-technical-report) | Overall framework description and results on MLE-bench                                                                                                                              |
| [R&D-Agent-Quant Release](#deep-application-in-diverse-scenarios) | Apply R&D-Agent to quant trading                                                                                                                                                 |
| MLE-Bench Results Released         | R&D-Agent currently leads as the [top-performing machine learning engineering agent](#-the-best-machine-learning-engineering-agent) on MLE-bench                                                            |
| Support LiteLLM Backend            | We now fully support **[LiteLLM](https://github.com/BerriAI/litellm)** as our default backend for integration with multiple LLM providers.                                                                  |
| General Data Science Agent         | [Data Science Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)                                                                                                                  |
| Kaggle Scenario release          | We release **[Kaggle Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)**, try the new features!                                                                                    |
| Official WeChat group release       | We created a WeChat group, welcome to join! (🗪[QR Code](https://github.com/microsoft/RD-Agent/issues/880))                                                                                                   |
| Official Discord release           | We launch our first chatting channel in Discord (🗪[![Chat](https://img.shields.io/badge/chat-discord-blue)](https://discord.gg/ybQ97B6Jjy))                                                                 |
| First release                      | **R&D-Agent** is released on GitHub                                                                                                                                                                         |

---

## Demo: Data Science Agent Preview

<div align="center">
    <a href="https://rdagent.azurewebsites.net/" target="_blank">
        <img src="docs/_static/demo.png" alt="Watch the demo" width="80%">
    </a>
</div>
<br>
<div align="center">
      <img src="docs/_static/scen.png" alt="Our focused scenario" style="width:80%; ">
</div>

R&D-Agent aims to automate the most critical and valuable aspects of the industrial R&D process, and we begin with focusing on the data-driven scenarios to streamline the development of models and data. 
Methodologically, we have identified a framework with two key components: 'R' for proposing new ideas and 'D' for implementing them.
We believe that the automatic evolution of R&D will lead to solutions of significant industrial value.

<!-- Tag Cloud -->
R&D is a very general scenario. The advent of R&D-Agent can be your
- 💰 **Automatic Quant Factory** ([🎥Demo Video](https://rdagent.azurewebsites.net/factor_loop)|[▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s))
- 🤖 **Data Mining Agent:** Iteratively proposing data & models ([🎥Demo Video 1](https://rdagent.azurewebsites.net/model_loop)|[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s)) ([🎥Demo Video 2](https://rdagent.azurewebsites.net/dmm)|[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4))  and implementing them by gaining knowledge from data.
- 🦾 **Research Copilot:** Auto read research papers ([🎥Demo Video](https://rdagent.azurewebsites.net/report_model)|[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKQ7o)) / financial reports ([🎥Demo Video](https://rdagent.azurewebsites.net/report_factor)|[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)) and implement model structures or building datasets.
- 🤖 **Kaggle Agent:** Auto Model Tuning and Feature Engineering([🎥Demo Video Coming Soon...]()) and implementing them to achieve more in competitions.
- ...

You can click the links above to view the demo. We're continuously adding more methods and scenarios to the project to enhance your R&D processes and boost productivity. 

Additionally, you can take a closer look at the examples in our **[🖥️ Live Demo](https://rdagent.azurewebsites.net/)**.


---

## Quick Start

### Prerequisites

*   Linux Operating System
*   Docker
*   Python 3.10 or 3.11
*   .env file to configure your LLM API settings

### 1. Install RD-Agent

```bash
# Create a Conda Environment
conda create -n rdagent python=3.10
conda activate rdagent

# Install RD-Agent
pip install rdagent
```

### 2. Configure your .env file

Choose one of the options below.

**Option 1: Using LiteLLM (Default)**
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
  If your environment configuration is complete, please execute the following commands to check if your configuration is valid. This step is necessary.
  ```bash
  rdagent health_check
  ```

### 3. Run a Demo

*   **Automated Quantitative Trading & Iterative Factors Model Joint Evolution:**

    ```bash
    rdagent fin_quant
    ```

*   **Automated Quantitative Trading & Iterative Factors Evolution:**

    ```bash
    rdagent fin_factor
    ```

*   **Automated Quantitative Trading & Iterative Model Evolution:**

    ```bash
    rdagent fin_model
    ```

*   **Automated Quantitative Trading & Factors Extraction from Financial Reports:**

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

*   **Automated Kaggle Model Tuning & Feature Engineering**: <br />
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

### 4. Monitor the Application Results

```bash
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

*   About the `data_science` parameter: If you want to see the logs of the data science scenario, set the `data_science` parameter to `True`; otherwise set it to `False`.

*   Although port 19899 is not commonly used, but before you run this demo, you need to check if port 19899 is occupied. If it is, please change it to another port that is not occupied.

  You can check if a port is occupied by running the following command.

  ```sh
  rdagent health_check --no-check-env --no-check-docker
  ```

---

## 🏭 Scenarios

### 🎯 Goal: Agent for Data-driven R&D

Our project aims to build an agent to automate data-driven R&D.  This involves:

*   **Extracting Information:** Reading reports and papers to extract key formulas, features, and model descriptions.
*   **Implementation:**  Translating extracted information into runnable code, with an evolving process for performance improvement.
*   **Idea Generation:** Proposing new ideas based on existing knowledge and observations.

### 📈 Scenarios/Demos

R&D-Agent acts as both a 🦾 Copilot and a 🤖 Agent in model implementation and data building:

| Scenario/Target | Model Implementation                   | Data Building                                                                      |
| --              | --                                     | --                                                                                 |
| **💹 Finance**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop)[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s) |  🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s) <br/>   🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)  |
| **🩺 Medical**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm)[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4) | -                                                                                  |
| **🏭 General**      | 🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model)[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKa7o) <br/> 🤖 Auto Kaggle Model Tuning   | 🤖Auto Kaggle feature Engineering |

*   **[RoadMap](https://rdagent.readthedocs.io/en/latest/scens/data_science.html#roadmap)**:  Explore new features in the Kaggle scenario.

### Successful Explorations

Find more demos here: [successful explorations](https://github.com/SunsetWolf/rdagent_resource/releases/download/demo_traces/demo_traces.zip).

More scenario details can be found in the **[📖readthedocs_scen](https://rdagent.readthedocs.io/en/latest/scens/catalog.html)**.

---

## ⚙️ Framework

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

Our framework addresses the need for automation in data science R&D:

| Research Area | Paper/Work List                                       |
| ------------- | ----------------------------------------------------- |
| **Benchmark the R&D abilities** | [Benchmark](#benchmark)                                                                                                                                |
| **Idea proposal:** Explore new ideas or refine existing ones          | [Research](#research)                                                                                                                                |
| **Ability to realize ideas:** Implement and execute ideas          | [Development](#development)                                                                                                                                |

Our core goal is to build an evolving R&D system, allowing agents to learn and improve.

---

## 📃 Paper/Work List

### Overall Technical Report

*   [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)

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

*   [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)

```BibTeX
@misc{chen2024datacentric,
    title={Towards Data-Centric Automatic R&D},
    author={Haotian Chen and Xinjie Shen and Zeqi Ye and Wenjun Feng and Haoxue Wang and Xiao Yang and Xu Yang and Weiqing Liu and Jiang Bian},
    year={2024},
    eprint={2404.11276},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

### 🔍 Research

Our framework focuses on continuous hypothesis generation, verification, and feedback loops.

### 🛠️ Development

*   [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)

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

*   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

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

---

## 🤝 Contributing

We welcome contributions to enhance R&D-Agent. See the [Contributing Guide](CONTRIBUTING.md).

### Guidelines

*   Report issues and bugs.
*   Suggest new features.
*   Improve documentation.
*   Submit code contributions through pull requests.

<img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>

<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

*Note:  Some internal contributions prior to the open-source release are not reflected in the public commit history.*

---

## ⚖️ Legal Disclaimer

*The RD-agent is provided “as is” without warranty. It is designed to facilitate research and development and is not intended for financial investment advice. Users must independently assess risks and comply with all applicable laws.*

---

**<a href="https://github.com/microsoft/RD-Agent"> 🚀 Explore R&D-Agent on GitHub </a>**