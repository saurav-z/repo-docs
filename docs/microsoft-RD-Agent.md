<h1 align="center">R&D-Agent: Revolutionizing Machine Learning Engineering with AI 🚀</h1>

R&D-Agent is a groundbreaking open-source project from Microsoft that leverages AI to automate and accelerate the machine learning research and development process, offering a powerful agent to streamline your workflows. Explore the original repository [here](https://github.com/microsoft/RD-Agent).

<div align="center">
  <img src="docs/_static/logo.png" alt="RA-Agent logo" style="width:70%; ">
</div>

## 🔑 Key Features

*   **Automated Machine Learning Engineering**: Automate the most critical and valuable aspects of the R&D process, streamlining model and data development.
*   **Multi-Agent Framework**: Leverage the power of multiple agents, each designed for specific tasks (Research and Development) to coordinate and optimize strategies.
*   **Data-Centric Approach**:  Focus on improving the development of models and data.
*   **Versatile Applications**: Applicable across diverse data-driven scenarios, including finance, medical research, and general data science tasks.
*   **MLE-Bench Leader**: R&D-Agent is a top-performing machine learning engineering agent on the MLE-bench benchmark.
*   **Comprehensive Documentation**: Detailed documentation, demos, and a live UI for easy exploration.
*   **Modular and Extensible**:  Designed for easy integration, and to build upon the framework.
*   **Open Source**: Get involved, contribute, and shape the future of AI-driven development.

## ✨ What is R&D-Agent?

R&D-Agent aims to automate the R&D process in data science and machine learning. The framework centers on two key components:  'R' (Research/Propose new ideas) and 'D' (Development/Implement). This automated evolution of R&D aims to deliver solutions with significant industrial value.

## 🌟 Key Highlights: MLE-Bench & Quant Finance

*   **MLE-Bench Performance**: R&D-Agent currently leads as the top-performing machine learning engineering agent on MLE-bench! The Agent achieved impressive results.
    *   **R&D-Agent o1-preview**: [Results](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
    *   **R&D-Agent o3(R)+GPT-4.1(D)**: [Results](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)
*   **R&D-Agent-Quant: Data-Centric Financial Research**:  R&D-Agent(Q) is the first of its kind, focusing on multi-agent frameworks designed for the automation of the full-stack research and development of quantitative strategies.

## 🚀 Quick Start

Get started with R&D-Agent quickly with these simple steps:

1.  **Environment**: Currently supports Linux.
2.  **Docker Installation**: Make sure you have Docker installed.
3.  **Conda Environment**: Create a conda environment (Python 3.10 or 3.11 recommended).
    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```
4.  **Installation**: Install R&D-Agent from PyPI:
    ```bash
    pip install rdagent
    ```
    For developers: Install from source.
5.  **Configuration**:  Set your preferred LLM and embedding models through LiteLLM or the deprecated setting.  See the README for detailed examples using OpenAI, Azure OpenAI, and DeepSeek.
6.  **Health Check**: Verify your setup.
    ```bash
    rdagent health_check
    ```
7.  **Run the Demo**: Explore scenarios through the Live Demo links below.

## 🏭 Scenarios & Demos

Explore R&D-Agent's capabilities through various scenarios and demos:

*   **Finance**:
    *   Automated Quantitative Trading & Iterative Factors Model Joint Evolution:  `rdagent fin_quant`
    *   Automated Quantitative Trading & Iterative Factors Evolution:  `rdagent fin_factor`
    *   Automated Quantitative Trading & Iterative Model Evolution:  `rdagent fin_model`
    *   Automated Quantitative Trading & Factors Extraction from Financial Reports: `rdagent fin_factor_report`
*   **Medical**: Automated Medical Prediction Model Evolution
*   **General Data Science**:
    *   Automated Model Research & Development Copilot: `rdagent general_model`
    *   Automated Kaggle Model Tuning & Feature Engineering: `rdagent data_science --competition tabular-playground-series-dec-2021`

Explore the **[🖥️ Live Demo](https://rdagent.azurewebsites.net/)** and demo videos for interactive examples.

*   **[🎥 Automated Quant Factory](https://rdagent.azurewebsites.net/factor_loop)**
*   **[🎥 Data Mining Agent: Iteratively Proposing Data & Models](https://rdagent.azurewebsites.net/model_loop)**
*   **[🎥 Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor)**
*   **[🎥 Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model)**

## ⚙️ Framework & Architecture

R&D-Agent is built on a framework that emphasizes automation and continuous improvement.

<div align="center">
    <img src="docs/_static/Framework-RDAgent.png" alt="Framework-RDAgent" width="85%">
</div>

The core research areas of the framework include:
*   **Benchmark:** Towards Data-Centric Automatic R&D
*   **Idea Proposal:**  Explore new ideas or refine existing ones
*   **Idea Realization:** Collaborative Evolving Strategy for Automatic Data-Centric Development

## 📃 Papers & Publications

Explore the foundational research behind R&D-Agent.

*   **Overall Technical Report:**  [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)
*   **Benchmark:** [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)
*   **Development:** [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)
*   **Deep Application in Diverse Scenarios:** [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

## 🤝 Contributing

Help shape the future of R&D-Agent!  Review the [Contributing Guide](CONTRIBUTING.md) for details on how to get involved.

<img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>
<a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

## ⚖️ Legal Disclaimer

Please review the legal disclaimer provided in the original README.

---

**[Get Started with R&D-Agent](https://github.com/microsoft/RD-Agent) and revolutionize your machine learning R&D!**