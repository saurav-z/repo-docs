<h1 align="center">
  <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:70%;">
  RD-Agent: Automate Your Data-Driven R&D
</h1>

<p align="center">
  <a href="https://github.com/microsoft/RD-Agent">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/microsoft/RD-Agent?style=social">
  </a>
</p>

<p align="center">
  <b>RD-Agent empowers AI to automate and accelerate the entire data-driven R&D process, offering cutting-edge capabilities for machine learning engineering and quantitative finance.</b>
</p>

<div align="center">
  <a href="https://rdagent.azurewebsites.net" target="_blank">🖥️ Live Demo</a> |
  <a href="https://rdagent.azurewebsites.net/factor_loop" target="_blank">🎥 Demo Video</a> |
  <a href="https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR" target="_blank">▶️ YouTube</a> |
  <a href="https://rdagent.readthedocs.io/en/latest/index.html" target="_blank">📖 Documentation</a> |
  <a href="https://aka.ms/RD-Agent-Tech-Report" target="_blank">📄 Tech Report</a> |
  <a href="#paperwork-list">📃 Papers</a> |
  <a href="https://discord.gg/ybQ97B6Jjy">💬 Discord</a>
</div>

---

## Key Features

*   🤖 **Automated R&D:** Automates critical R&D tasks in data-driven scenarios.
*   🥇 **Top Performance on MLE-bench:** Leads in the [MLE-bench](https://github.com/openai/mle-bench) benchmark for machine learning engineering.
*   📈 **Data-Centric Focus:** Built for data-centric factor and model joint optimization.
*   💡 **Iterative Improvement:** Employs an iterative approach to learn from feedback and improve results.
*   💻 **Versatile Applications:** Applicable to finance, medical research, and more.
*   📖 **Comprehensive Documentation:** Detailed documentation to get you started.
*   ⚙️ **Easy Installation:** Simple setup with Docker and Conda.
*   🌐 **Community Support:** Active Discord community for discussions and assistance.

---

## MLE-Bench Leaderboard

RD-Agent excels on the MLE-bench, demonstrating superior performance in machine learning engineering tasks.  See the latest results:

| Agent                       | Low == Lite (%) | Medium (%) | High (%) | All (%)    |
|-----------------------------|-----------------|------------|----------|------------|
| R&D-Agent o1-preview        | 48.18 ± 2.49    | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1  |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21    | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview             | 34.3 ± 2.4      | 8.8 ± 1.1    | 10.0 ± 1.9   | 16.9 ± 1.1  |

**Notes:**
*   **O3(R)+GPT-4.1(D)**: Leveraging a Research Agent (o3) with a Development Agent (GPT-4.1).
*   **AIDE o1-preview**:  The previous leading result on MLE-bench.
*   Averaged and standard deviation results for R&D-Agent o1-preview are based on 5 seeds, while R&D-Agent o3(R)+GPT-4.1(D) results are based on 6 seeds.
*   MLE-Bench categorizes competitions based on complexity: **Low==Lite** (under 2 hours for an experienced ML engineer), **Medium** (2-10 hours), and **High** (over 10 hours).

Explore detailed runs:
*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

## Data Science Agent Preview

Check out a demo video:
<div align="center">
<img src="https://github.com/user-attachments/assets/3eccbecb-34a4-4c81-bce4-d3f8862f7305" alt="Data Science Agent Demo" width="80%">
</div>

## Applications & Scenarios

RD-Agent is designed to excel in various data-driven R&D scenarios:

*   **Automatic Quant Factory:**  Automated quantitative trading strategy development ([🎥Demo Video](https://rdagent.azurewebsites.net/factor_loop) | [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s))
*   **Data Mining Agent:** Iteratively proposes and implements data & models ([🎥Demo Video 1](https://rdagent.azurewebsites.net/model_loop) | [▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s)) ([🎥Demo Video 2](https://rdagent.azurewebsites.net/dmm) | [▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4))
*   **Research Copilot:** Automatically reads research papers ([🎥Demo Video](https://rdagent.azurewebsites.net/report_model) | [▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKQ7o)) and financial reports ([🎥Demo Video](https://rdagent.azurewebsites.net/report_factor) | [▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c))
*   **Kaggle Agent:** Automated model tuning and feature engineering (Demo Coming Soon)
*   **... and more!**

Explore the [Live Demo](https://rdagent.azurewebsites.net/) to experience these features.

<div align="center">
    <a href="https://rdagent.azurewebsites.net/" target="_blank">
        <img src="docs/_static/demo.png" alt="Watch the demo" width="80%">
    </a>
</div>

---

## Quick Start

1.  **System Requirements:** Linux only.
2.  **Docker Installation:** Ensure Docker is installed.  See the [official Docker documentation](https://docs.docker.com/engine/install/).
3.  **Conda Environment:**
    ```bash
    conda create -n rdagent python=3.10  # or python=3.11
    conda activate rdagent
    ```
4.  **Install RD-Agent:**
    ```bash
    pip install rdagent
    ```
    Or, for development:
    ```bash
    git clone https://github.com/microsoft/RD-Agent
    cd RD-Agent
    make dev
    ```
5.  **Health Check:**
    ```bash
    rdagent health_check --no-check-env
    ```
6.  **Configuration:**  Configure your LLM and embedding models. See the example below or refer to the [documentation](https://rdagent.readthedocs.io/en/latest/installation_and_configuration.html).

    **Using LiteLLM (Default)**: Supports multiple LLM providers.

    **Option 1: Unified API Base**
    ```bash
    cat << EOF  > .env
    CHAT_MODEL=gpt-4o
    EMBEDDING_MODEL=text-embedding-3-small
    OPENAI_API_BASE=<your_unified_api_base>
    OPENAI_API_KEY=<replace_with_your_openai_api_key>
    EOF
    ```

    **Option 2: Separate API Bases**
    ```bash
    cat << EOF  > .env
    # CHAT MODEL:
    CHAT_MODEL=gpt-4o
    OPENAI_API_BASE=<your_chat_api_base>
    OPENAI_API_KEY=<replace_with_your_openai_api_key>

    # EMBEDDING MODEL:
    EMBEDDING_MODEL=litellm_proxy/BAAI/bge-large-en-v1.5
    LITELLM_PROXY_API_KEY=<replace_with_your_siliconflow_api_key>
    LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1
    EOF
    ```

    **Configuration Example: DeepSeek Setup** (working example):

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

    ```bash
    REASONING_THINK_RM=True
    ```

7.  **Verify Configuration:** Run health check:
    ```bash
    rdagent health_check
    ```
8.  **Run a Demo:**  Choose a scenario (each command represents a demo):

    *   Automated Quantitative Trading (Qlib):
        ```bash
        rdagent fin_quant
        ```
    *   Automated Quantitative Trading (Factor Evolution):
        ```bash
        rdagent fin_factor
        ```
    *   Automated Quantitative Trading (Model Evolution):
        ```bash
        rdagent fin_model
        ```
    *   Extract Factors from Financial Reports:
        ```bash
        wget https://github.com/SunsetWolf/rdagent_resource/releases/download/reports/all_reports.zip
        unzip all_reports.zip -d git_ignore_folder/reports
        rdagent fin_factor_report --report_folder=git_ignore_folder/reports
        ```
    *   Automated Model Research Copilot:
        ```bash
        rdagent general_model "https://arxiv.org/pdf/2210.09789"  # Replace with your paper
        ```
    *   Automated Data Science application.
        ```bash
        wget https://github.com/SunsetWolf/rdagent_resource/releases/download/ds_data/arf-12-hours-prediction-task.zip
        unzip arf-12-hours-prediction-task.zip -d ./git_ignore_folder/ds_data/
        dotenv set DS_LOCAL_DATA_PATH "$(pwd)/git_ignore_folder/ds_data"
        dotenv set DS_CODER_ON_WHOLE_PIPELINE True
        dotenv set DS_IF_USING_MLE_DATA False
        dotenv set DS_SAMPLE_DATA_BY_LLM False
        dotenv set DS_SCEN rdagent.scenarios.data_science.scen.DataScienceScen
        rdagent data_science --competition arf-12-hours-prediction-task
        ```
    *   Automated Kaggle Model Tuning & Feature Engineering:
        ```bash
        mkdir -p ./git_ignore_folder/ds_data
        dotenv set DS_LOCAL_DATA_PATH "$(pwd)/git_ignore_folder/ds_data"
        dotenv set DS_CODER_ON_WHOLE_PIPELINE True
        dotenv set DS_IF_USING_MLE_DATA True
        dotenv set DS_SAMPLE_DATA_BY_LLM True
        dotenv set DS_SCEN rdagent.scenarios.data_science.scen.KaggleScen
        rdagent data_science --competition tabular-playground-series-dec-2021
        ```
9.  **Monitor Results:**
    ```bash
    rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
    ```

---

## Scenarios

Explore a range of data-driven scenarios:
*   **[Finance](https://rdagent.readthedocs.io/en/latest/scens/quant_agent_fin.html):**  Automated trading strategies, factor extraction.
*   **[Medical](https://rdagent.readthedocs.io/en/latest/scens/data_science.html):**  Automated model development.
*   **[General Research](https://rdagent.readthedocs.io/en/latest/scens/data_science.html):**  Paper reading, code implementation.
*   **[Kaggle](https://rdagent.readthedocs.io/en/latest/scens/data_science.html#roadmap):**  Automated model tuning and feature engineering.

---

## Framework Overview

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

RD-Agent's framework focuses on automating the R&D process through:

*   **Benchmark:** Evaluating R&D capabilities.
*   **Idea Proposal:** Generating new ideas or refining existing ones.
*   **Idea Realization:** Implementing and executing ideas.

See the [Documentation](https://rdagent.readthedocs.io/) for more details.

---

## <a name="paperwork-list"></a> Paper/Work List

### Overall Technical Report

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

### Benchmark

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

### Research

*   [See above sections for demo examples](https://rdagent.azurewebsites.net)

### Development

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

### Deep Application in Diverse Scenarios

*   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)
    ```bibtex
    @misc{li2025rdagentquant,
        title={R\&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization},
        author={Yuante Li and Xu Yang and Xiao Yang and Minrui Xu and Xisen Wang and Xisen Wang and Weiqing Liu and Jiang Bian},
        year={2025},
        eprint={2505.15155},
        archivePrefix={arXiv},
        primaryClass={cs.AI}
    }
    ```

---

## Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) for details.  Please ensure your code passes CI checks before submitting a pull request.

<p align="center">
  <img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>
  <a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
  </a>
</p>

---

## Legal Disclaimer

*The RD-Agent is provided “as is”...*  (See the original for the full disclaimer.)

---

[Back to Top](#)