<h1 align="center">🤖 RD-Agent: The Premier Machine Learning Engineering Agent </h1>

<p align="center">
    <a href="https://github.com/microsoft/RD-Agent">
        <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:40%;">
    </a>
</p>

<p align="center">
   **RD-Agent is a groundbreaking, AI-powered framework designed to automate and revolutionize the data-driven R&D process, leading the charge in machine learning engineering automation.**
</p>

<p align="center">
  <a href="https://rdagent.azurewebsites.net" target="_blank">🖥️ Live Demo</a> |
  <a href="https://rdagent.azurewebsites.net/factor_loop" target="_blank">🎥 Demo Video</a> |
  <a href="https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR" target="_blank">▶️YouTube</a>   |
  <a href="https://rdagent.readthedocs.io/en/latest/index.html" target="_blank">📖 Documentation</a> |
  <a href="https://aka.ms/RD-Agent-Tech-Report" target="_blank">📄 Tech Report</a> |
  <a href="#-paperwork-list"> 📃 Papers </a> |
  <a href="https://discord.gg/ybQ97B6Jjy" target="_blank">💬 Discord</a>
</p>

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
[![Documentation Status](https://readthedocs.org/projects/rdagent/badge/?version=latest)](https://rdagent.readthedocs.io/en/latest/?badge=latest)
[![Readthedocs Preview](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2505.14738-00ff00.svg)](https://arxiv.org/abs/2505.14738)

## Key Features

*   **Automated R&D:** Streamlines the development of models and data, automating critical aspects of the R&D process.
*   **Multi-Agent Framework:** Utilizes a multi-agent architecture for collaborative and efficient task execution.
*   **Data-Centric Approach:** Focuses on data-driven scenarios to optimize model and data evolution.
*   **Leading Performance:** Consistently outperforms other agents in the [MLE-bench](https://github.com/openai/mle-bench) benchmark.
*   **Versatile Applications:** Applicable across diverse domains, including finance, medical, and general research.
*   **Extensive Documentation:** Detailed documentation and examples to get you started quickly.

## 🥇 RD-Agent: The Leader in Machine Learning Engineering

RD-Agent is revolutionizing machine learning engineering, consistently achieving top results on the [MLE-bench](https://github.com/openai/mle-bench) benchmark.

Here's how RD-Agent excels:

| Agent | Low == Lite (%) | Medium (%) | High (%) | All (%) |
|---------|--------|-----------|---------|----------|
| R&D-Agent o1-preview | 48.18 ± 2.49 | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21 | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview | 34.3 ± 2.4 | 8.8 ± 1.1 | 10.0 ± 1.9 | 16.9 ± 1.1 |

**Learn more:**
*   **R&D-Agent o1-preview detailed runs:** [https://aka.ms/RD-Agent_MLE-Bench_O1-preview](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   **R&D-Agent o3(R)+GPT-4.1(D) detailed runs:** [https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

## 💰 Applications of RD-Agent:

RD-Agent can be your:

*   💰 **Automatic Quant Factory** ([🎥Demo Video](https://rdagent.azurewebsites.net/factor_loop)|[▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s))
*   🤖 **Data Mining Agent:** Iteratively proposing data & models ([🎥Demo Video 1](https://rdagent.azurewebsites.net/model_loop)|[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s)) ([🎥Demo Video 2](https://rdagent.azurewebsites.net/dmm)|[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4))  and implementing them by gaining knowledge from data.
*   🦾 **Research Copilot:** Auto read research papers ([🎥Demo Video](https://rdagent.azurewebsites.net/report_model)|[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKa7o)) / financial reports ([🎥Demo Video](https://rdagent.azurewebsites.net/report_factor)|[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)) and implement model structures or building datasets.
*   🤖 **Kaggle Agent:** Auto Model Tuning and Feature Engineering ([🎥Demo Video Coming Soon...]()) and implementing them to achieve more in competitions.
*   ...

## 🚀 Getting Started

### Prerequisites

*   Linux OS
*   Docker (Installation instructions: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/))
*   Conda (Python 3.10 or 3.11 recommended)

### Installation

1.  **Create and activate a Conda environment:**

    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```

2.  **Install RD-Agent from PyPI (for users):**

    ```bash
    pip install rdagent
    ```

3.  **Or, install from source (for developers):**

    ```bash
    git clone https://github.com/microsoft/RD-Agent
    cd RD-Agent
    make dev
    ```
    More details in the [development setup](https://rdagent.readthedocs.io/en/latest/development.html).

### ⚙️ Configuration

Configure your Chat and Embedding models using LiteLLM, with multiple setup options:

*   **Option 1: Unified API base:**
    ```bash
    cat << EOF  > .env
    CHAT_MODEL=gpt-4o
    EMBEDDING_MODEL=text-embedding-3-small
    OPENAI_API_BASE=<your_unified_api_base>
    OPENAI_API_KEY=<replace_with_your_openai_api_key>
    EOF
    ```

*   **Option 2: Separate API bases:**
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

*   **DeepSeek Setup Example:**
    ```bash
    cat << EOF  > .env
    CHAT_MODEL=deepseek/deepseek-chat
    DEEPSEEK_API_KEY=<your_deepseek_api_key>

    EMBEDDING_MODEL=litellm_proxy/BAAI/bge-m3
    LITELLM_PROXY_API_KEY=<replace_with_your_siliconflow_api_key>
    LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1
    EOF
    ```

    *   Set `REASONING_THINK_RM=True` if your reasoning models use  `<think>` tags.

###  Health Check

Verify your configuration:

```bash
rdagent health_check
```

### 🏃 Running the Demo

Run the Automated Quantitative Trading & Iterative Factors Model Joint Evolution demo:

```bash
rdagent fin_quant
```

Other demos available, refer to the [README](https://github.com/microsoft/RD-Agent)

### 🖥️ Monitoring Results

View run logs:

```bash
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

## 🌐 Scenarios and Use Cases

RD-Agent is designed for data-driven R&D, with applications in these key areas:

### 💹 Finance
*   Automated Quant Trading & Iterative Factor/Model Joint Evolution

### 🩺 Medical
*   Automated Medical Prediction Model Evolution

### 🏭 General
*   Automated Model Research & Development Copilot (Paper Reading & Implementation)
*   Automated Kaggle Model Tuning and Feature Engineering

## 📚 Framework Overview

RD-Agent utilizes a cutting-edge framework to automate R&D processes, with a focus on:

*   **Benchmarking R&D abilities.**
*   **Idea Proposal:** Generating and refining ideas.
*   **Idea Realization:** Implementing and executing ideas.

## 📃 Paperwork List

*   **Overall Technical Report:** [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)
*   **Benchmark:** [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)
*   **Research:** Details on idea generation and real-world verification.
*   **Development:** [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)
*   **Deep Application:** [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

## 🤝 Contributing

We welcome contributions!  See the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## ⚖️ Legal Disclaimer

*Please read the legal disclaimer provided in the original README.*

---

**[Explore the RD-Agent repository on GitHub](https://github.com/microsoft/RD-Agent) and join the future of machine learning engineering!**