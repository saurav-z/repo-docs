<h1 align="center">🤖 R&D-Agent: Automating Data-Driven R&D with AI</h1>

R&D-Agent is a cutting-edge framework designed to automate and revolutionize the research and development process, empowering you to build AI-driven solutions more efficiently.  Explore the power of AI in automating the R&D cycle; [check out the original repo](https://github.com/microsoft/RD-Agent).

<div align="center">
  <img src="docs/_static/logo.png" alt="RA-Agent logo" style="width:60%; ">
</div>

## 🚀 Key Features:

*   **Automated R&D:** Streamlines the entire research and development pipeline, from idea generation to implementation.
*   **Multi-Agent Framework:** Leverages a coordinated system of agents to handle diverse R&D tasks.
*   **Data-Centric Approach:** Focuses on data-driven scenarios to optimize model and data development.
*   **Kaggle Agent:**  Enhances your performance on Kaggle competitions with automated model tuning and feature engineering.
*   **Quant Finance Focus:**  RD-Agent(Q) offers a first-of-its-kind, data-centric, multi-agent framework for automated quantitative strategy development.
*   **Flexible LLM Integration:** Supports various LLM providers, including LiteLLM integration for streamlined access.
*   **Comprehensive Demos:**  Experience RD-Agent's capabilities through live demos and demo videos, showcasing model and data generation in action.

## 📊 Leading Performance on MLE-bench

R&D-Agent is a top performer on the MLE-bench benchmark, demonstrating its effectiveness in machine learning engineering tasks.  Explore the details:

| Agent                     | Low == Lite (%) | Medium (%) | High (%) | All (%)     |
| :------------------------ | :-------------: | :--------: | :-------: | :----------: |
| R&D-Agent o1-preview      |  48.18 ± 2.49   | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1  |
| R&D-Agent o3(R)+GPT-4.1(D) |  51.52 ± 6.21   | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview           |  34.3 ± 2.4    |  8.8 ± 1.1  |  10.0 ± 1.9 | 16.9 ± 1.1  |

*   **Detailed Runs:** [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview), [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)
*   **MLE-bench Guide:** [Running ML Engineering via MLE-bench](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)

## ⚙️ Use Cases and Applications

R&D-Agent is versatile and adaptable to various data-driven scenarios.

*   **Automated Quant Factory:** Generate factors and models to optimize trading strategies. [🎥Demo Video](https://rdagent.azurewebsites.net/factor_loop)|[▶️YouTube](https://www.youtube.com/watch?v=X4DK2QZKaKY&t=6s))
*   **Data Mining Agent:**  Iteratively explores and proposes data & models. ([🎥Demo Video 1](https://rdagent.azurewebsites.net/model_loop)|[▶️YouTube](https://www.youtube.com/watch?v=dm0dWL49Bc0&t=104s)) ([🎥Demo Video 2](https://rdagent.azurewebsites.net/dmm)|[▶️YouTube](https://www.youtube.com/watch?v=VIaSTZuoZg4))
*   **Research Copilot:** Automatically reads and implements research papers and financial reports. ([🎥Demo Video](https://rdagent.azurewebsites.net/report_model)|[▶️YouTube](https://www.youtube.com/watch?v=BiA2SfdKa7o)) / financial reports ([🎥Demo Video](https://rdagent.azurewebsites.net/report_factor)|[▶️YouTube](https://www.youtube.com/watch?v=ECLTXVcSx-c))
*   **Kaggle Agent:** Streamlines your Kaggle competition experience with model tuning and feature engineering.

## 🌟 Get Started

### ⚡ Quick Start
1.  **Prerequisites:** Linux OS is currently required. Ensure Docker is installed and your user has permissions to run Docker commands.
2.  **Environment Setup:** Create a Conda environment and activate it.
3.  **Installation:** Install the `rdagent` package using `pip install rdagent`.  For development, install from source using `make dev`.
4.  **Configuration:** Configure your API keys in the `.env` file, including Chat and Embedding model settings. LiteLLM is supported.
5.  **Health Check:** Verify your setup with `rdagent health_check`.
6.  **Run a Demo:** Execute the demo scenarios with command.
    *   `rdagent fin_quant`
    *   `rdagent fin_factor`
    *   `rdagent fin_model`
    *   `rdagent fin_factor_report --report_folder=<Your financial reports folder path>`
    *   `rdagent general_model  "https://arxiv.org/pdf/2210.09789"`
    *   `rdagent data_science --competition <your competition name>`
7.  **Monitor Results:** Use `rdagent ui` to view logs and track progress.

## 🌐 Resources

*   **💻 Live Demo:** [https://rdagent.azurewebsites.net/](https://rdagent.azurewebsites.net/)
*   **📖 Documentation:** [https://rdagent.readthedocs.io/en/latest/index.html](https://rdagent.readthedocs.io/en/latest/index.html)
*   **📄 Technical Report:** [https://aka.ms/RD-Agent-Tech-Report](https://aka.ms/RD-Agent-Tech-Report)
*   **📝 Papers:**
    *   [R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution](https://arxiv.org/abs/2505.14738)
    *   [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276)
    *   [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)
    *   [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)

*   **💬 Discord:** [![Chat](https://img.shields.io/badge/chat-discord-blue)](https://discord.gg/ybQ97B6Jjy)

## 🤝 Contributing

We welcome your contributions! Please review the [Contributing Guide](CONTRIBUTING.md) before submitting any pull requests.