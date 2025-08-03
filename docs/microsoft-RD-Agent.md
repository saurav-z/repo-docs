<h1 align="center">RD-Agent: Automating Data-Driven R&D with AI</h1>

<p align="center">
  <a href="https://github.com/microsoft/RD-Agent">
    <img src="docs/_static/logo.png" alt="RD-Agent Logo" width="300">
  </a>
  <br>
  <i>RD-Agent empowers AI agents to autonomously automate the R&D process, leading to breakthroughs in machine learning engineering.</i>
  <br>
  <a href="https://rdagent.azurewebsites.net" target="_blank">🖥️ Live Demo</a> |
  <a href="https://rdagent.azurewebsites.net/factor_loop" target="_blank">🎥 Demo Video</a> |
  <a href="https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR" target="_blank">▶️YouTube</a> |
  <a href="https://rdagent.readthedocs.io/en/latest/index.html" target="_blank">📖 Documentation</a> |
  <a href="https://aka.ms/RD-Agent-Tech-Report" target="_blank">📄 Tech Report</a> |
  <a href="#-paperwork-list"> 📃 Papers </a> |
  <a href="https://discord.gg/ybQ97B6Jjy" target="_blank"> 💬 Discord</a>
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
[![Documentation Status](https://readthedocs.org/projects/rdagent/badge/?version=latest)](https://rdagent.readthedocs.io/en/latest/?badge=latest)
[![Readthedocs Preview](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2505.14738-00ff00.svg)](https://arxiv.org/abs/2505.14738)

## Key Features

*   **Automated Machine Learning Engineering:** Automates tasks from data mining to model building and evaluation.
*   **Multi-Agent Framework:**  Employs a multi-agent system to mimic the R&D process.
*   **Data-Centric Approach:**  Focuses on data-driven scenarios to streamline model and data development.
*   **Integration with Leading LLMs:** Support for LiteLLM and other LLM providers.
*   **Open Source:** Explore and contribute to the code on GitHub.

## Overview

R&D-Agent is a cutting-edge framework designed to automate the most critical aspects of industrial R&D, focusing initially on data-driven scenarios.  It utilizes a multi-agent system, with components for 'R' (Research) and 'D' (Development), to iteratively propose, implement, and refine solutions.  This innovative approach aims to significantly boost productivity in R&D processes.

##  🚀  Getting Started

### System Requirements

*   RD-Agent currently only supports Linux.

### Installation

1.  **Docker Installation:** Ensure Docker is installed and correctly configured.
2.  **Conda Environment:**
    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```
3.  **Install RD-Agent:**
    ```bash
    pip install rdagent
    ```
    or for development:
    ```bash
    git clone https://github.com/microsoft/RD-Agent
    cd RD-Agent
    make dev
    ```

### Configuration

1.  **Environment Setup:** Configure your environment variables (API keys, model names) using .env files. Example configurations are provided in the documentation and the Quick Start guide.
2.  **Health Check:** Verify your configuration:
    ```bash
    rdagent health_check
    ```

### Running the Application

Select and run the desired demo using the `rdagent` command. Examples include:

*   `rdagent fin_quant` (Automated Quantitative Trading)
*   `rdagent fin_factor` (Automated Factor Evolution)
*   `rdagent fin_model` (Automated Model Evolution)
*   `rdagent fin_factor_report --report_folder=<Your financial reports folder path>` (Financial Report Analysis)
*   `rdagent general_model <Your paper URL>` (Paper/Report Implementation)
*   `rdagent data_science --competition <your competition name>` (Kaggle/Medical Scenario)

See detailed scenario setups in the [documentation](https://rdagent.readthedocs.io/en/latest/scens/catalog.html).

### Monitoring Results

Monitor the application logs and results using the UI:

```bash
rdagent ui --port 19899 --log_dir <your log folder> --data_science <True or False>
```

## 🌟  Key Advantages

*   **Leading Performance on MLE-bench:** R&D-Agent achieves top performance on the MLE-bench benchmark.
*   **Automated R&D:**  Significantly reduces manual effort in research and development.
*   **Flexible and Extensible:**  Easily adapts to various data-driven scenarios.
*   **Open Source and Collaborative:**  Contribute to the project and shape its future.

## 🏆 Performance Highlights:  Top-Performing Agent on MLE-bench

R&D-Agent has demonstrated exceptional performance, consistently outperforming other agents on the [MLE-bench](https://github.com/openai/mle-bench) benchmark, which evaluates AI agents on machine learning engineering tasks using datasets from 75 Kaggle competitions.

| Agent                         | Low == Lite (%) | Medium (%) | High (%) | All (%)   |
| ----------------------------- | --------------- | ----------- | -------- | --------- |
| R&D-Agent o1-preview          | 48.18 ± 2.49    | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D)   | 51.52 ± 6.21    | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview               | 34.3 ± 2.4      | 8.8 ± 1.1   | 10.0 ± 1.9 | 16.9 ± 1.1 |

**Notes:**

*   Detailed results available at:  [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview) and  [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41).
*   Refer to the MLE-Bench documentation to categorize the competition levels.

## 🏭 Scenarios

RD-Agent is designed for these data-driven industrial scenarios.

### **Financial R&D**

*   Automated quant strategy development.
*   Automated factor & model joint evolution
*   Auto reports reading & implementation

### **Medical R&D**

*   Medical Model evolution.

### **General Data Science**

*   Automated paper reading & implementation
*   Automated Kaggle competition model tuning and feature engineering

## 📰 News

*   **[Technical Report Release](#overall-technical-report):** Overall framework description and results on MLE-bench.
*   **[R&D-Agent-Quant Release](#deep-application-in-diverse-scenarios):** Application of R&D-Agent to quant trading.
*   **MLE-Bench Results Released:**  R&D-Agent is currently a top-performing machine learning engineering agent.
*   **LiteLLM Support:**  Fully supports [LiteLLM](https://github.com/BerriAI/litellm) as the backend for multiple LLM providers.
*   **Data Science Agent and Kaggle Scenario Release:**  Explore the new features.
*   **Community Channels:** Official Discord and WeChat group releases.

## 🤝 Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) for details.

## ⚖️ Legal Disclaimer
... (Include your legal disclaimer)

## 🔗 Additional Resources

*   [Documentation](https://rdagent.readthedocs.io/en/latest/index.html)
*   [Live Demo](https://rdagent.azurewebsites.net/)
*   [YouTube Channel](https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR)
*   [Technical Report](https://aka.ms/RD-Agent-Tech-Report)
*   [Papers](#-paperwork-list)
*   [Original Repo](https://github.com/microsoft/RD-Agent)