<h1 align="center">RD-Agent: Automating Data-Driven R&D with LLMs</h1>

<div align="center">
  <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:50%;">
</div>

RD-Agent is a cutting-edge framework designed to **automate the entire machine learning engineering process, from research to deployment**, achieving state-of-the-art results on MLE-bench! 🚀  Learn more about this innovative project on its [GitHub repository](https://github.com/microsoft/RD-Agent).

<p align="center">
  <a href="https://rdagent.azurewebsites.net" target="_blank">🖥️ Live Demo</a> |
  <a href="https://rdagent.azurewebsites.net/factor_loop" target="_blank">🎥 Demo Video</a> |
  <a href="https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR" target="_blank">▶️YouTube</a>  |
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
[![Documentation Status](https://readthedocs.org/projects/rdagent/badge/?version=latest)](https://rdagent.readthedocs.io/en/latest/?badge=latest)
[![Readthedocs Preview](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2505.14738-00ff00.svg)](https://arxiv.org/abs/2505.14738)


## Key Features

*   **Automated R&D:** Automates the critical and valuable aspects of industrial R&D, streamlining the development of models and data.
*   **Multi-Agent Framework:** A data-centric, multi-agent framework designed for full-stack research and development.
*   **Leading Performance on MLE-bench:** Currently leads as the top-performing machine learning engineering agent on MLE-bench.
*   **Data-Centric Approach:** Focused on data-driven scenarios, automating the process of model and data development.
*   **Flexible and Extensible:** Supports various scenarios, including finance, medical, and general research, with continuous expansion.
*   **Ease of Use:** Provides straightforward installation and configuration, with comprehensive documentation and demos.
*   **Open Source and Collaborative:** Welcomes contributions and suggestions, promoting a collaborative environment for improvement.

## Performance Highlights

RD-Agent consistently demonstrates superior performance across various benchmarks:

###  Leading on MLE-bench

| Agent | Low == Lite (%) | Medium (%) | High (%) | All (%) |
|---------|--------|-----------|---------|----------|
| R&D-Agent o1-preview | 48.18 ± 2.49 | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21 | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview | 34.3 ± 2.4 | 8.8 ± 1.1 | 10.0 ± 1.9 | 16.9 ± 1.1 |

### R&D-Agent(Q) Quantitative Finance Performance

Extensive experiments in real stock markets show that, at a cost under $10, RD-Agent(Q) achieves approximately 2× higher ARR than benchmark factor libraries while using over 70% fewer factors. It also surpasses state-of-the-art deep time-series models under smaller resource budgets.

Detailed results can be found on [MLE-bench Results](https://aka.ms/RD-Agent_MLE-Bench_O1-preview) and [R&D-Agent(Q) Detailed Runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41).

## Quick Start

### Prerequisites
*   **Linux:** RD-Agent currently only supports Linux.
*   **Docker:** Ensure Docker is installed. [Official Docker page](https://docs.docker.com/engine/install/) for installation.
*   **Conda:** Create and activate a Conda environment, Python 3.10 or 3.11 recommended.

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
3.  **Install the R&D-Agent**
    ```bash
    pip install rdagent
    ```
    Or for developers, install from source:
    ```bash
    make dev
    ```

### Configuration

Configure your Chat and Embedding Models.

**LiteLLM Configuration (Recommended):**
```bash
cat << EOF  > .env
# CHAT MODEL:
CHAT_MODEL=gpt-4o
# EMBEDDING MODEL:
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=<replace_with_your_openai_api_key>
OPENAI_API_BASE=<your_openai_api_base>
EOF
```

**Example: DeepSeek Setup**

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

### Run the Application

Run the desired demo using the example commands.  For instance:

```bash
rdagent fin_quant
```

## 🏭 Scenarios

R&D-Agent currently supports the following scenarios:

*   **Finance:** Automated Quantitative Trading and Iterative Factor/Model Evolution.
*   **Medical:** Automated Medical Prediction Model Evolution.
*   **General:** Automated Paper Reading and Implementation, Kaggle Model Tuning and Feature Engineering.

See the **[📖 Documentation](https://rdagent.readthedocs.io/en/latest/scens/catalog.html)** for setup instructions.

## 🤝 Contributing

We welcome contributions! Refer to the [Contributing Guide](CONTRIBUTING.md) for details.

## 📃 Paper/Work List

*   **Overall Technical Report:** [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)
*   **Benchmark:** [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)
*   **Development:** [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)
*   **Deep Application in Diverse Scenarios:** [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

## ⚖️ Legal Disclaimer

*The RD-Agent is provided “as is”, without warranty.*  Users must independently assess risks, comply with all laws/regulations and assume all liability.

---

**Get started with RD-Agent today and revolutionize your data-driven R&D processes!**