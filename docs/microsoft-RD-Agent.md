<h1 align="center">🤖 RD-Agent: Automate Your Machine Learning Engineering with AI</h1>

<p align="center">
  <a href="https://github.com/microsoft/RD-Agent">
    <img src="docs/_static/logo.png" alt="RD-Agent Logo" width="30%">
  </a>
</p>

**Unlock the power of automated machine learning engineering with RD-Agent, your AI-powered research and development assistant.**

RD-Agent is a groundbreaking framework developed by Microsoft designed to automate and optimize the R&D process for machine learning engineering tasks. Leveraging the power of Large Language Models (LLMs), RD-Agent empowers you to accelerate your ML projects, from data exploration to model deployment.

**➡️ [Explore the RD-Agent Repo](https://github.com/microsoft/RD-Agent)**

## 🚀 Key Features

*   **Leading Performance:** Currently leads as the top-performing machine learning engineering agent on MLE-bench, demonstrating superior results.
*   **Automated R&D:** Automates key aspects of the R&D process, including idea generation, implementation, and evaluation.
*   **Multi-Agent Framework:** Coordinates multiple agents to streamline the development of quantitative strategies via coordinated factor-model co-optimization.
*   **Data-Centric Approach:** Focuses on automating the full-stack research and development of quantitative strategies.
*   **Versatile Applications:**
    *   Automated Quant Factory
    *   Data Mining Agent
    *   Research Copilot
    *   Kaggle Agent
*   **Easy to Use:** Provides docker installation and python environment for a quick start.

## 🌟 What's New?

*   **[Overall Technical Report Release](https://arxiv.org/abs/2505.14738):** Framework Description, Benchmarks & Results
*   **[R&D-Agent-Quant Release](https://arxiv.org/abs/2505.15155):** Deep Application in Diverse Scenarios
*   **[MLE-Bench Results Released:** R&D-Agent currently leads as the top-performing machine learning engineering agent on MLE-bench
*   **[LiteLLM Backend Support](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#-configuration):** Full support for LiteLLM as a backend for easy integration with multiple LLM providers.
*   **[Data Science Agent](https://rdagent.readthedocs.io/en/latest/scens/data_science.html):** Explore the power of RD-Agent on Data Science Problems
*   **Kaggle Agent Release:**  Try the new features!
*   **Discord and WeChat Community:** Join the community!

## 📈 Benchmarking & Results

RD-Agent demonstrates superior performance on MLE-bench, a comprehensive benchmark for evaluating AI agents in machine learning engineering:

| Agent                | Low == Lite (%) | Medium (%) | High (%) | All (%)    |
| -------------------- | --------------- | ---------- | -------- | ---------- |
| R&D-Agent o1-preview | 48.18 ± 2.49    | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R&D-Agent o3(R)+GPT-4.1(D) | 51.52 ± 6.21    | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview      | 34.3 ± 2.4      | 8.8 ± 1.1   | 10.0 ± 1.9 | 16.9 ± 1.1 |

Detailed results are available:

*   [R&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview)
*   [R&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41)

For running R&D-Agent on MLE-bench, refer to **[MLE-bench Guide: Running ML Engineering via MLE-bench](https://rdagent.readthedocs.io/en/latest/scens/data_science.html)**

## 🛠️ Quick Start

1.  **Environment:** Ensure you have a Linux environment.
2.  **Docker Installation:** Install Docker following the [official Docker documentation](https://docs.docker.com/engine/install/).
3.  **Conda Environment:** Create and activate a Conda environment: `conda create -n rdagent python=3.10`, then `conda activate rdagent`.
4.  **Installation:**  `pip install rdagent`
5.  **Configuration**: Configure the environment.
    *  **Use LiteLLM (Default)**:  Configure CHAT_MODEL and EMBEDDING_MODEL and the appropriate API key in a `.env` file.
        * Option 1: Unified API Base:
            ```bash
            cat << EOF  > .env
            # Set to any model supported by LiteLLM.
            CHAT_MODEL=gpt-4o 
            EMBEDDING_MODEL=text-embedding-3-small
            # Configure unified API base
            OPENAI_API_BASE=<your_unified_api_base>
            OPENAI_API_KEY=<replace_with_your_openai_api_key>
            ```
        * Option 2: Separate API Bases:
           ```bash
            cat << EOF  > .env
            # Set to any model supported by LiteLLM.
            # Configure separate API bases for chat and embedding

            # CHAT MODEL:
            CHAT_MODEL=gpt-4o
            OPENAI_API_BASE=<your_chat_api_base>
            OPENAI_API_KEY=<replace_with_your_openai_api_key>

            # EMBEDDING MODEL:
            # TAKE siliconflow as an example, you can use other providers.
            # Note: embedding requires litellm_proxy prefix
            EMBEDDING_MODEL=litellm_proxy/BAAI/bge-large-en-v1.5
            LITELLM_PROXY_API_KEY=<replace_with_your_siliconflow_api_key>
            LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1
            ```
    *  **DeepSeek Setup:**  Example configuration.
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
        ```

6.  **Health Check:** `rdagent health_check` (Ensure configuration is valid.)
7.  **Run the application:**
    *   `rdagent fin_quant`: Automated Quantitative Trading
    *   `rdagent fin_factor`: Automated Quantitative Trading & Iterative Factors Evolution
    *   `rdagent fin_model`: Automated Quantitative Trading & Iterative Model Evolution
    *   `rdagent fin_factor_report --report_folder=<Your financial reports folder path>`
    *   `rdagent general_model <Your paper URL>`
    *   `rdagent data_science --competition arf-12-hours-prediction-task` (Medical)
    *   `rdagent data_science --competition tabular-playground-series-dec-2021` (Kaggle)
8.  **Monitor Results:**  `rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>`

## 🔗 Demos & Resources

*   [🖥️ Live Demo](https://rdagent.azurewebsites.net/)
*   [🎥 Demo Video](https://rdagent.azurewebsites.net/factor_loop)
*   [📖 Documentation](https://rdagent.readthedocs.io/en/latest/index.html)
*   [📄 Tech Report](https://aka.ms/RD-Agent-Tech-Report)
*   [📃 Papers](#-paperwork-list)
*   [Successful explorations](https://github.com/SunsetWolf/rdagent_resource/releases/download/demo_traces/demo_traces.zip)

## 🤝 Contribute

Contributions are welcome!  Refer to the [Contributing Guide](CONTRIBUTING.md).

## ⚖️ Legal

See the [Legal disclaimer](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#-%E5%88%A9%E5%AE%B3%E7%9B%B8%E5%85%B3) for important legal information.