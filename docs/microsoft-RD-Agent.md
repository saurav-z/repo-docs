<h1 align="center">RD-Agent: Automate Your Machine Learning Engineering with AI</h1>

<p align="center">
    <a href="https://github.com/microsoft/RD-Agent" target="_blank">
        <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:200px;">
    </a>
    <br>
    <b>RD-Agent empowers you to automate the R&D process for data-driven AI solutions.</b>
    <br>
    <a href="https://rdagent.azurewebsites.net" target="_blank">🖥️ Live Demo</a> |
    <a href="https://rdagent.azurewebsites.net/factor_loop" target="_blank">🎥 Demo Video</a> |
    <a href="https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR" target="_blank">▶️YouTube</a> |
    <a href="https://rdagent.readthedocs.io/en/latest/index.html" target="_blank">📖 Documentation</a> |
    <a href="https://aka.ms/RD-Agent-Tech-Report" target="_blank">📄 Tech Report</a> |
    <a href="#-paperwork-list"> 📃 Papers </a> |
    <a href="https://discord.gg/ybQ97B6Jjy"> 💬 Discord</a>
</p>

---

RD-Agent is a cutting-edge framework designed to automate the core R&D tasks in data-driven scenarios, leading to significant advancements in machine learning engineering and quantitative finance. Explore the [original repo](https://github.com/microsoft/RD-Agent) for the latest updates.

## Key Features

*   🚀 **Automated R&D:** Automates the entire R&D lifecycle, from idea generation to implementation.
*   🧠 **Multi-Agent Framework:** Features a multi-agent architecture for collaborative task execution.
*   📊 **Top-Performing:** Proven performance on the MLE-bench benchmark.
*   📈 **Data-Centric Focus:** Specifically designed for data-centric R&D in quantitative finance.
*   🛠️ **Flexible Scenarios:** Adaptable to various applications, including finance, medical research, and Kaggle competitions.
*   🌐 **Open Source:** Open-source and welcomes community contributions.

## MLE-bench Performance

RD-Agent has achieved impressive results on the MLE-bench benchmark, demonstrating its capabilities in real-world machine learning engineering.

| Agent                     | Low == Lite (%) | Medium (%) | High (%) | All (%)  |
| ------------------------- | --------------- | ---------- | -------- | -------- |
| R&D-Agent o1-preview      | 48.18 ± 2.49    | 8.95 ± 2.36 | 18.67 ± 2.98  | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21    | 7.89 ± 3.33  | 16.67 ± 3.65  | 22.45 ± 2.45 |
| AIDE o1-preview           | 34.3 ± 2.4      | 8.8 ± 1.1   | 10.0 ± 1.9  | 16.9 ± 1.1 |

**Detailed results:**
*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

## Use Cases and Demos

RD-Agent offers a variety of applications, including:

*   **Automated Quant Factory:** [🎥Demo Video](https://rdagent.azurewebsites.net/factor_loop) | [▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s)
*   **Data Mining Agent:**  [🎥Demo Video 1](https://rdagent.azurewebsites.net/model_loop) | [▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s)
    [🎥Demo Video 2](https://rdagent.azurewebsites.net/dmm) | [▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4)
*   **Research Copilot:** [🎥Demo Video](https://rdagent.azurewebsites.net/report_model) | [▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKQ7o)
    / Financial Reports [🎥Demo Video](https://rdagent.azurewebsites.net/report_factor) | [▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c)
*   **Kaggle Agent:** (Coming Soon)
*   ...

Explore the **[🖥️ Live Demo](https://rdagent.azurewebsites.net/)** to see these scenarios in action.

## Quick Start

### Prerequisites
*   Linux Environment
*   Docker (for many scenarios)
*   Python 3.10 or 3.11

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/microsoft/RD-Agent
    cd RD-Agent
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```

3.  **Install RD-Agent:**
    ```bash
    pip install rdagent
    ```
    For developers:
    ```bash
    make dev
    ```

### Configuration
Configure your LLM and embedding models using LiteLLM or direct OpenAI / Azure OpenAI API settings. See the original README for comprehensive details, including DeepSeek configurations and environment variable setup.

### Running the Application
Run the available demos by using the `rdagent` CLI tool (see the original README for example commands).

### Monitoring the Application Results
Use the UI for the results by running
```bash
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

## Framework Overview

RD-Agent's research is organized around the areas of benchmarking R&D abilities, idea proposal, and the ability to realize ideas, which help evolve R&D capabilities.

See the full framework in the original README.

## Paper/Work List

### Overall Technical Report
*   [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)

### Benchmark
*   [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)

### Research
*   The research focuses on continuous hypothesis proposal, verification, and feedback to improve R&D.

### Development
*   [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)

### Deep Application in Diverse Scenarios
*   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

---

## Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md).

## Legal Disclaimer

Read the legal disclaimer in the original README.

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