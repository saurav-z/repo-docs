<h1 align="center">🤖 R&D-Agent: Automating Data-Driven Innovation</h1>

<p align="center">
  <a href="https://github.com/microsoft/RD-Agent" target="_blank">
    <img src="docs/_static/logo.png" alt="RD-Agent Logo" width="30%">
  </a>
</p>

<p align="center">
  <i>Unlock the future of machine learning engineering with R&D-Agent, the first data-centric, multi-agent framework for automating the R&D process.</i>
</p>

<div align="center">
  <a href="https://rdagent.azurewebsites.net/" target="_blank">🖥️ Live Demo</a> |
  <a href="https://rdagent.azurewebsites.net/factor_loop" target="_blank">🎥 Demo Video</a> |
  <a href="https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR" target="_blank">▶️YouTube</a> |
  <a href="https://rdagent.readthedocs.io/en/latest/index.html" target="_blank">📖 Documentation</a> |
  <a href="https://aka.ms/RD-Agent-Tech-Report" target="_blank">📄 Tech Report</a> |
  <a href="#-paperwork-list"> 📃 Papers </a>
  <br>
  <img src="https://img.shields.io/github/stars/microsoft/RD-Agent?style=social" alt="GitHub stars">
</div>

---

## Key Features

*   **Automated ML Engineering:** Automate the entire R&D lifecycle, from idea generation to code implementation, for faster iteration and improved results.
*   **Data-Centric Approach:** Focuses on data-driven scenarios, optimizing models and data for superior performance.
*   **Multi-Agent Framework:** Leverages a collaborative framework with distinct "R" (Research) and "D" (Development) agents for efficient task execution.
*   **Leading Performance:** Achieve state-of-the-art results in various benchmarks, including leading performance on the MLE-bench.
*   **Real-World Applications:** Applied successfully in financial modeling, medical research, and Kaggle competitions.
*   **Modular and Extensible:** Easily integrate with LLM providers like LiteLLM and experiment with new scenarios.

---

## Why Choose R&D-Agent?

R&D-Agent is designed to significantly streamline your machine learning R&D processes, making it:

*   **More Efficient:** Automate repetitive tasks, accelerating the R&D cycle.
*   **More Innovative:** Explore new ideas and approaches with less manual effort.
*   **More Effective:** Achieve superior results through data-centric optimization and continuous improvement.

---

## 🚀 Quick Start

Get started with R&D-Agent in just a few steps!

### 🛠️ Installation

R&D-Agent currently supports Linux.

#### 🐳 Docker Installation
Ensure that Docker is installed on your system by following the [official Docker installation instructions](https://docs.docker.com/engine/install/). Then, make sure that the current user has permission to run Docker commands without `sudo` by executing `docker run hello-world`.

#### 🐍 Conda Environment

1.  Create a Conda environment (Python 3.10 or 3.11 recommended):

    ```bash
    conda create -n rdagent python=3.10
    ```

2.  Activate the environment:

    ```bash
    conda activate rdagent
    ```

#### 📦 Install R&D-Agent

*   **For Users:** Install from PyPI:

    ```bash
    pip install rdagent
    ```

*   **For Developers:** Install from source (for the latest features or contribution):

    ```bash
    git clone https://github.com/microsoft/RD-Agent
    cd RD-Agent
    make dev
    ```

### 💊 Health Check

Verify your installation:

```bash
rdagent health_check --no-check-env
```

### ⚙️ Configuration

1.  Configure Chat and Embedding Models:

    R&D-Agent uses LiteLLM by default. Set your `CHAT_MODEL` and `EMBEDDING_MODEL` in a `.env` file.

    *   **Option 1: Unified API (Recommended):**

        ```bash
        cat << EOF  > .env
        CHAT_MODEL=gpt-4o
        EMBEDDING_MODEL=text-embedding-3-small
        OPENAI_API_BASE=<your_unified_api_base>
        OPENAI_API_KEY=<your_openai_api_key>
        EOF
        ```

    *   **Option 2: Separate API Bases:**

        ```bash
        cat << EOF  > .env
        CHAT_MODEL=gpt-4o
        OPENAI_API_BASE=<your_chat_api_base>
        OPENAI_API_KEY=<your_openai_api_key>

        EMBEDDING_MODEL=litellm_proxy/BAAI/bge-large-en-v1.5
        LITELLM_PROXY_API_KEY=<your_siliconflow_api_key>
        LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1
        EOF
        ```

    *   **DeepSeek Setup Example:**

        ```bash
        cat << EOF  > .env
        CHAT_MODEL=deepseek/deepseek-chat
        DEEPSEEK_API_KEY=<your_deepseek_api_key>

        EMBEDDING_MODEL=litellm_proxy/BAAI/bge-m3
        LITELLM_PROXY_API_KEY=<your_siliconflow_api_key>
        LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1
        EOF
        ```

    **Important:** If using reasoning models, set: `REASONING_THINK_RM=True`

2.  Verify Configuration:

    ```bash
    rdagent health_check
    ```

### 🚀 Run the Application

Select a scenario and run the corresponding command:

*   **Automated Quantitative Trading & Iterative Factor Model Joint Evolution:**

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
    rdagent fin_factor_report --report_folder=<Your financial reports folder path>
    ```

*   **Automated Model Research & Development Copilot:**

    ```bash
    rdagent general_model "https://arxiv.org/pdf/2210.09789"
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

View run logs:

```bash
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

---

## 🏭 Scenarios

R&D-Agent supports a variety of scenarios, including:

*   **Finance:** Automated Quant Trading and Factors Extraction.
*   **Medical:** Automated Model Evolution.
*   **General:** Automated Paper Reading and Kaggle Competition Integration

See [Scenarios](https://rdagent.readthedocs.io/en/latest/scens/catalog.html) for more details.

---

## ⚙️ Framework Overview

The R&D-Agent framework automates the data science R&D process:

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

The framework's key areas of research are:

*   **Benchmark the R&D abilities**: [Benchmark](#benchmark)
*   **Idea Proposal:** [Research](#research)
*   **Implementation of Ideas:** [Development](#development)

---

## 📃 Paper/Work List

### Overall Technical Report
*   [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)

### 📊 Benchmark
*   [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)

### 🔍 Research
*   In a data mining expert's daily research and development process, they propose a hypothesis (e.g., a model structure like RNN can capture patterns in time-series data), design experiments (e.g., finance data contains time-series and we can verify the hypothesis in this scenario), implement the experiment as code (e.g., Pytorch model structure), and then execute the code to get feedback (e.g., metrics, loss curve, etc.). The experts learn from the feedback and improve in the next iteration.

### 🛠️ Development
*   [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)

### Deep Application in Diverse Scenarios
*   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

---

## 🤝 Contributing

We welcome contributions to R&D-Agent!  See the [Contributing Guide](CONTRIBUTING.md) for more details.

---

## ⚖️ Legal Disclaimer

*The RD-agent is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. The RD-agent is aimed to facilitate research and development process in the financial industry and not ready-to-use for any financial investment or advice. Users shall independently assess and test the risks of the RD-agent in a specific use scenario, ensure the responsible use of AI technology, including but not limited to developing and integrating risk mitigation measures, and comply with all applicable laws and regulations in all applicable jurisdictions. The RD-agent does not provide financial opinions or reflect the opinions of Microsoft, nor is it designed to replace the role of qualified financial professionals in formulating, assessing, and approving finance products. The inputs and outputs of the RD-agent belong to the users and users shall assume all liability under any theory of liability, whether in contract, torts, regulatory, negligence, products liability, or otherwise, associated with use of the RD-agent and any inputs and outputs thereof.*

---

**[Visit the RD-Agent GitHub Repository](https://github.com/microsoft/RD-Agent)**