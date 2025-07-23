<h1 align="center">🤖 R&D-Agent: Automating Data-Driven Research and Development</h1>

<div align="center">
  <img src="docs/_static/logo.png" alt="RA-Agent logo" style="width:50%; max-width:400px;">
  <p><b>R&D-Agent is a cutting-edge, AI-powered agent designed to revolutionize data-driven research and development by automating key aspects of the R&D process.</b></p>
  <a href="https://github.com/microsoft/RD-Agent" target="_blank">
    <img src="https://img.shields.io/github/stars/microsoft/RD-Agent?style=social" alt="Stars">
  </a>
</div>

---

R&D-Agent accelerates your data-driven R&D by automating tasks like idea generation, code implementation, and iterative improvement, leading to faster innovation and higher-quality solutions. This repository provides the source code for R&D-Agent, along with demonstrations, documentation, and resources.

**Key Features:**

*   **Automated R&D:** Automates key R&D processes, from idea generation to implementation and evaluation.
*   **Multi-Agent Framework:**  A data-centric, multi-agent framework designed to automate the full-stack research and development of quantitative strategies via coordinated factor-model co-optimization.
*   **Data-Driven Approach:**  Focuses on data-driven scenarios for model and data development.
*   **Real-World Applications:** Demonstrates value in quantitative finance, medical research, Kaggle competitions, and general research tasks.
*   **Extensible and Customizable:**  Easily adaptable to new scenarios and research areas.
*   **Integration with LiteLLM**: Fully supports [LiteLLM](https://github.com/BerriAI/litellm) as the default backend for integration with multiple LLM providers, offering flexibility in model selection.

**Key Metrics & Results:**

R&D-Agent consistently leads as the top-performing machine learning engineering agent on the MLE-bench benchmark, demonstrating robust performance in real-world ML engineering scenarios.

*   [MLE-Bench Results](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#-%EF%B8%8F-the-best-machine-learning-engineering-agent-): See comparative agent performance on the MLE-bench machine learning engineering benchmark.
*   [R&D-Agent-Quant Results](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#-deep-application-in-diverse-scenarios): Shows a 2x increase in Average Return Rate (ARR) at a cost of under $10.

**Quick Links:**

*   🖥️ [Live Demo](https://rdagent.azurewebsites.net)
*   🎥 [Demo Video](https://rdagent.azurewebsites.net/factor_loop) | [YouTube](https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR)
*   📖 [Documentation](https://rdagent.readthedocs.io/en/latest/index.html)
*   📄 [Technical Report](https://aka.ms/RD-Agent-Tech-Report)
*   📃 [Papers](#-paperwork-list)
*   💻 [Source Code on GitHub](https://github.com/microsoft/RD-Agent)

**Getting Started:**

*   **[Quick Start Instructions](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#-quick-start):**  Comprehensive instructions for installing and running the agent.  Includes Docker setup, environment configuration, and running example scenarios.

**Scenarios & Applications:**

R&D-Agent is designed for diverse data-driven applications:

*   **Finance:** Automated quantitative trading strategy development.
*   **Medical Research:** Automated model iteration and improvement for medical prediction tasks.
*   **Kaggle Competitions:** Automated Model Tuning and Feature Engineering.
*   **General Research:** Automated paper reading, model implementation, and idea generation.

**[See detailed Scenario/Demo documentation](https://rdagent.readthedocs.io/en/latest/scens/catalog.html)**

**Framework Overview:**

R&D-Agent is built on a framework that focuses on:

*   **[Benchmark the R&D abilities](#benchmark)**
*   **[Idea proposal](#research):** Research & Explore new ideas
*   **[Ability to realize ideas](#development):** Implement and execute ideas

**Framework Diagram:**
<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

**[See the Framework Diagram](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#-%EF%B8%8F-framework)**

**Join the Community:**
*   <a href="https://discord.gg/ybQ97B6Jjy" target="_blank">
      <img src="https://img.shields.io/badge/chat-discord-blue" alt="Discord">
    </a>

**Contributions are welcomed!**

*   [Contributing Guide](CONTRIBUTING.md)

---

**[See the full paper list.](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#-paperwork-list)**

**Legal disclaimer:**

*   [Read the Legal Disclaimer](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#-legal-disclaimer)

<!-- Badges -->
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
```

Key improvements and SEO considerations:

*   **Clear, Concise Introduction:**  Immediately states the value proposition ("revolutionize data-driven research...") in a way that's easily understood.
*   **SEO-Optimized Headings:** Uses clear, keyword-rich headings (e.g., "Key Features," "Getting Started," "Scenarios & Applications").
*   **Keyword Integration:** Naturally incorporates relevant keywords such as "AI agent," "machine learning engineering," "data-driven R&D," "automated," "benchmark," and "quantitative finance."
*   **Concise Bullet Points:**  Highlights key features with concise and impactful bullet points.
*   **Call to Action:** Encourages interaction (e.g., "Join the Community," "Contributions are welcomed!").
*   **Links to Key Resources:** Prominent links to the live demo, documentation, and technical report.
*   **Structured Content:** Uses markdown to organize the information logically.
*   **Visual Appeal:** Includes a logo, social badge, and other visual elements to make the README more engaging.
*   **Clear Language:** Avoids overly technical jargon.
*   **Emphasis on Results:**  Highlights the MLE-bench performance and other key results to build credibility.
*   **Clear Navigation:** Uses anchor links to improve navigation.
*   **Complete and Comprehensive:**  Includes all the important information from the original README in a more organized and accessible format.
*   **Concise and Readable:** Focuses on readability and getting the core information across quickly.