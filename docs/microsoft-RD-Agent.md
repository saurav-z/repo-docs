# RD-Agent: Automate Machine Learning Engineering & Data Science with AI Agents

**Unleash the power of AI to automate machine learning engineering and data science tasks, streamlining your research and development with the RD-Agent ([Original Repo](https://github.com/microsoft/RD-Agent)).**

<div align="center">
  <img src="docs/_static/logo.png" alt="RD-Agent Logo" style="width:70%;">
</div>

## Key Features

*   **Automated ML Engineering:**  Orchestrate the entire machine learning workflow, from data analysis to model deployment.
*   **Multi-Agent Framework:** Leverage a collaborative system of AI agents to handle diverse aspects of R\&D, including idea generation, implementation, and optimization.
*   **Data-Centric Approach:** Focus on optimizing data and model selection for superior results.
*   **Cutting-Edge Performance:** Achieve state-of-the-art results on benchmarks like MLE-bench, surpassing other leading agents.
*   **Flexible and Extensible:** Easily integrate with existing tools and adapt to different project requirements.
*   **Real-World Application:** Proven performance in various data-driven scenarios, including quantitative finance and data science tasks.

## Benchmarking and Performance

RD-Agent consistently outperforms in the leading machine learning engineering benchmark, MLE-bench:

| Agent                       | Low == Lite (%) | Medium (%) | High (%) | All (%)  |
| --------------------------- | --------------- | ----------- | -------- | -------- |
| R\&D-Agent o1-preview        | 48.18 ± 2.49    | 8.95 ± 2.36 | 18.67 ± 2.98 | 22.4 ± 1.1 |
| R\&D-Agent o3(R)+GPT-4.1(D)  | 51.52 ± 6.21    | 7.89 ± 3.33 | 16.67 ± 3.65 | 22.45 ± 2.45 |
| AIDE o1-preview             | 34.3 ± 2.4      | 8.8 ± 1.1   | 10.0 ± 1.9 | 16.9 ± 1.1 |

*   Detailed performance runs are available online:  [R\&D-Agent o1-preview detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O1-preview) and [R\&D-Agent o3(R)+GPT-4.1(D) detailed runs](https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41).

## Scenarios and Demos

Explore RD-Agent's capabilities through diverse scenarios and demos, with more being actively developed.

*   **Finance:**  Automate quantitative trading strategies and factor model optimization:
    *   [Automated Quantitative Trading & Iterative Factors Model Joint Evolution](https://rdagent.azurewebsites.net/factor_loop)
    *   [Automated Quantitative Trading & Iterative Factors Evolution](https://rdagent.azurewebsites.net/factor_loop)
    *   [Automated Quantitative Trading & Iterative Model Evolution](https://rdagent.azurewebsites.net/model_loop)
    *   [Automated Quantitative Trading & Factors Extraction from Financial Reports](https://rdagent.azurewebsites.net/report_factor)
*   **Medical:** Accelerate medical model development:
    *   [Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm)
*   **General:**  Streamline model research and development.
    *   [Auto Paper Reading & Implementation](https://rdagent.azurewebsites.net/report_model)
    *   Automated Kaggle Model Tuning
    *   Automated Kaggle Feature Engineering

*   **[🖥️ Live Demo](https://rdagent.azurewebsites.net/)**

## Quick Start

### Prerequisites

*   **Linux:**  RD-Agent currently only supports Linux.
*   **Docker:**  Install [Docker](https://docs.docker.com/engine/install/) to run the demos.
*   **Conda:** Create and activate a Conda environment (Python 3.10 or 3.11 recommended):
    ```bash
    conda create -n rdagent python=3.10
    conda activate rdagent
    ```
*   **Install:**

    ```bash
    pip install rdagent
    ```
    or for development:
    ```bash
    git clone https://github.com/microsoft/RD-Agent
    cd RD-Agent
    make dev
    ```

*   **Health Check:**  Verify your setup.
    ```bash
    rdagent health_check --no-check-env
    ```

### Configuration

Set up your API keys for Chat and Embedding models (LiteLLM is the default). See the example below:

  > **🔥 Attention**: We now provide experimental support for **DeepSeek** models! You can use DeepSeek's official API for cost-effective and high-performance inference. See the configuration example below for DeepSeek setup.

```bash
cat << EOF  > .env
# Set to any model supported by LiteLLM.
# Configure unified API base
CHAT_MODEL=gpt-4o 
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_BASE=<your_unified_api_base>
OPENAI_API_KEY=<replace_with_your_openai_api_key>
```

### Running the Application

Choose a demo and run it:

*   Automated Quantitative Trading:  `rdagent fin_quant`
*   Automated Model Research:  `rdagent general_model "https://arxiv.org/pdf/2210.09789"`

### Monitoring Results
```bash
rdagent ui --port 19899 --log_dir <your log folder like "log/"> --data_science <True or False>
```

## Framework Overview

The core of RD-Agent lies in its R\&D framework.

*   **Benchmark the R\&D abilities:**  [Benchmark](https://arxiv.org/abs/2404.11276)
*   **Idea proposal:** [Research](https://rdagent.azurewebsites.net)
*   **Ability to realize ideas:** [Development](https://arxiv.org/abs/2407.18690)

More documents can be found in the **[📖 readthedocs](https://rdagent.readthedocs.io/)**.

## Papers and Resources

*   **[Technical Report](https://arxiv.org/abs/2505.14738):**  R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution
*   **[Benchmark](https://arxiv.org/abs/2404.11276):** Towards Data-Centric Automatic R&D
*   **[Research](https://rdagent.azurewebsites.net):**
*   **[Development](https://arxiv.org/abs/2407.18690):** Collaborative Evolving Strategy for Automatic Data-Centric Development
*   **[Deep Application in Diverse Scenarios](https://arxiv.org/abs/2505.15155):** R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization
*   **[📖 Documentation](https://rdagent.readthedocs.io/)**
*   **[📄 Tech Report](https://aka.ms/RD-Agent-Tech-Report)**
*   **[🖥️ Live Demo](https://rdagent.azurewebsites.net/)**
*   **[🎥 Demo Video](https://rdagent.azurewebsites.net/factor_loop)**
*   **[▶️YouTube](https://www.youtube.com/watch?v=JJ4JYO3HscM&list=PLALmKB0_N3_i52fhUmPQiL4jsO354uopR)**

## Contributing

We welcome contributions! See the [Contributing Guide](CONTRIBUTING.md) for details.

*   <img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>
*   <a href="https://github.com/microsoft/RD-Agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/RD-Agent&max=100&columns=15" />
</a>

## Legal

[Legal disclaimer](https://github.com/microsoft/RD-Agent?tab=readme-ov-file#%E2%9A%9B%EF%B8%8F-legal-disclaimer)