<h1 align="center">🚀 RD-Agent: Revolutionizing Data-Driven Research and Development with AI</h1>

<p align="center">
  RD-Agent is a cutting-edge, open-source framework designed to automate and accelerate the R&D process, offering a powerful AI-driven solution for data-centric challenges. <a href="https://github.com/microsoft/RD-Agent">Explore the code on GitHub</a>.
</p>

<div align="center">
  <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:50%;">
</div>

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

## ✨ Key Features

*   **Automated R&D:** Streamlines model and data development by automating critical R&D processes.
*   **Multi-Agent Framework:** Coordinates multiple AI agents to tackle complex data-driven tasks.
*   **Data-Centric Approach:** Focuses on extracting value from data by proposing, implementing, and evolving ideas.
*   **Versatile Applications:**
    *   Automated Quantitative Trading & Iterative Factor Model Joint Evolution
    *   Data Mining Agent for Iterative Model and Data Evolution
    *   Research Copilot for auto reading & implementation
    *   Automated Kaggle Model Tuning & Feature Engineering
*   **Leading Performance:** Ranked #1 machine learning engineering agent on the MLE-bench benchmark.

## 📊 MLE-Bench Performance

R&D-Agent demonstrates superior performance on the [MLE-bench](https://github.com/openai/mle-bench) benchmark.

| Agent | Low == Lite (%) | Medium (%) | High (%) | All (%) |
|---------|--------|-----------|---------|----------|
| R&D-Agent o1-preview | 48.18 ± 2.49 | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21 | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview | 34.3 ± 2.4 | 8.8 ± 1.1 | 10.0 ± 1.9 | 16.9 ± 1.1 |

**Detailed runs can be inspected online:**

*   [R&D-Agent o1-preview](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D)](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

## 📈 Scenarios & Demos

RD-Agent is designed to serve as both a 🦾 Copilot and a 🤖 Agent in various data-driven scenarios.

| Scenario/Target | Model Implementation                   | Data Building                                                                      |
| --              | --                                     | --                                                                                 |
| **💹 Finance**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop)[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s) |  🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s) <br/>   🦾 [Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)  |
| **🩺 Medical**      | 🤖 [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm)[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4) | -                                                                                  |
| **🏭 General**      | 🦾 [Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model)[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKa7o) <br/> 🤖 Auto Kaggle Model Tuning   | 🤖Auto Kaggle feature Engineering |

**Explore the Live Demo:** [🖥️ Live Demo](https://rdagent.azurewebsites.net/)

## 🚀 Quick Start

**Prerequisites:** Linux is required for RD-Agent to run. Docker is required for many scenarios.

**Installation:**

1.  **Docker Installation:** Follow instructions from the [official 🐳Docker page](https://docs.docker.com/engine/install/).  Make sure the current user can run Docker commands without `sudo`.
2.  **Conda Environment:**
    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```
3.  **Install RD-Agent:**
    *   **For Users:** `pip install rdagent`
    *   **For Developers:**
        ```bash
        git clone https://github.com/microsoft/RD-Agent
        cd RD-Agent
        make dev
        ```
4.  **Health Check:** Ensure your environment is set up correctly.
    ```bash
    rdagent health_check --no-check-env
    ```
5.  **Configuration:** Configure your Chat and Embedding models (e.g., OpenAI, DeepSeek) via `.env` file (see detailed instructions in the original README).  LiteLLM is supported.
6.  **Validate Configuration:**
    ```bash
    rdagent health_check
    ```
7.  **Run Demos:** Run demos using commands such as: `rdagent fin_quant`, `rdagent fin_factor`, `rdagent data_science --competition <your competition name>` (See the original README for detailed examples and the  [documentation](https://rdagent.readthedocs.io/en/latest/scens/catalog.html)).
8.  **Monitor Results:** Run the UI: `rdagent ui --port 19899 --log_dir <your log folder> --data_science <True or False>`

## 🏭 Scenarios

RD-Agent excels in various data-driven industrial scenarios.

## 📁 Framework

The core of RD-Agent lies in its multi-agent framework designed for automatic R&D. Key areas of research within the framework include: Benchmark, Idea Proposal and Ability to realize ideas.  For more details refer to the **[📖 readthedocs](https://rdagent.readthedocs.io/)**.

## 🤝 Contributing

We welcome contributions to the RD-Agent project. Refer to the [Contributing Guide](CONTRIBUTING.md).

## ⚖️ Legal Disclaimer

[Read the full legal disclaimer in the original README](https://github.com/microsoft/RD-Agent#-legal-disclaimer).

[Back to Top](#)