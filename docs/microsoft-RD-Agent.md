<h1 align="center">🤖 R&D-Agent: Automate Your Data-Driven Research & Development</h1>

<div align="center">
  <img src="docs/_static/logo.png" alt="R&D-Agent logo" style="width:50%; margin-bottom: 20px;">
</div>

R&D-Agent is an innovative AI agent designed to revolutionize the R&D process, leading to automated and efficient data-driven solutions. 

*   [🖥️ Live Demo](https://rdagent.azurewebsites.net/) |
*   [🎥 Demo Video](https://rdagent.azurewebsites.net/factor_loop) | [▶️YouTube](https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR)
*   [📖 Documentation](https://rdagent.readthedocs.io/en/latest/index.html) | [📄 Tech Report](https://aka.ms/RD-Agent-Tech-Report)
*   [📃 Papers](#-paperwork-list)

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

## 🚀 Key Features

*   **Automated R&D:** Streamlines the R&D process, focusing on data-driven scenarios.
*   **Data-Centric Approach:**  Designed to automate the development of models and data, focusing on the most valuable aspects of the industrial R&D process.
*   **Multi-Agent Framework:** Leverages a multi-agent system for coordinated research and development.
*   **Adaptable Framework:** Supports diverse data-driven scenarios (e.g., Finance, Medical, General).
*   **Continuous Improvement:**  Agents learn and evolve, constantly refining their capabilities.
*   **Integration with MLE-bench:** Leading machine-learning engineering agent on MLE-bench.
*   **Real-World Application:**  Achieves higher ARR with fewer factors in real stock markets.
*   **Extensive Documentation**: Rich documentation for a wide variety of use cases, including a [Quick Start](https://rdagent.readthedocs.io/en/latest/getting_started.html)

## 🥇 Leading Performance on MLE-Bench

R&D-Agent consistently demonstrates outstanding performance on the [MLE-bench](https://github.com/openai/mle-bench) benchmark:

| Agent                      | Low == Lite (%) | Medium (%) | High (%) | All (%) |
| -------------------------- | --------------- | ---------- | -------- | ------- |
| R&D-Agent o1-preview       | 48.18 ± 2.49    | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21    | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview            | 34.3 ± 2.4      | 8.8 ± 1.1   | 10.0 ± 1.9 | 16.9 ± 1.1 |

## 🏭 Scenarios & Use Cases

R&D-Agent is versatile and applicable across multiple domains:

*   **Finance:** Automated quantitative trading strategy development.
*   **Medical:**  Model development for medical predictions.
*   **General:** Automating paper reading, model implementation, and feature engineering.
*   **Kaggle Competition**: Automating model tuning & feature engineering.

## ⚡ Quick Start

Get started with R&D-Agent quickly:

1.  **Environment Setup:** R&D-Agent currently supports Linux.
2.  **Installation:** Install using `pip install rdagent`.
3.  **Configuration:** Configure your chat and embedding models (e.g., using LiteLLM).  
4.  **Run the application**: Use the commands to start your applications
5.  **Monitoring**: Monitor your runs through the UI.

For detailed instructions, see the full [Quickstart](https://rdagent.readthedocs.io/en/latest/getting_started.html) and [installation](https://rdagent.readthedocs.io/en/latest/installation_and_configuration.html) guides.

## ⚙️ Framework Overview

R&D-Agent is built on a robust framework designed for automating the R&D process:

*   **Idea Proposal:** Generate new ideas for model and factor improvement.
*   **Implementation:** Implement generated ideas through code generation and execution.
*   **Iteration & Learning:** Continuously improve by learning from results and feedback.

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

## 📃 Paper/Work List

Explore the research behind R&D-Agent:

*   **Overall Technical Report:** [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)
*   **Benchmark:** [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)
*   **Research:**  Focus on scientific research automation.
*   **Development:**  [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)
*   **Deep Application in Diverse Scenarios:** [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

## 🤝 Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) for details.

## ⚖️ Legal Disclaimer
See legal disclaimer at the end of the original README.

[Visit the original repository for more details](https://github.com/microsoft/RD-Agent).