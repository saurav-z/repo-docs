# 🤖 Youtu-Agent: Build Powerful Agents with Open-Source Models [🔗](https://github.com/TencentCloudADP/Youtu-agent)

Unleash the power of autonomous agents with Youtu-Agent, a flexible and high-performance framework delivering impressive results using open-source models.

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

## Key Features

*   ✅ **High Performance & Open-Source:** Achieves SOTA results on benchmarks like WebWalkerQA and GAIA using open-source models, reducing costs and promoting accessibility.
*   🛠️ **Practical Use Cases:**  Out-of-the-box support for diverse tasks, including data analysis, file processing, research, and video/podcast generation.
*   ⚙️ **Flexible Architecture:** Built on OpenAI's `openai-agents-python` for seamless integration with various LLMs (DeepSeek, gpt-oss, etc.), tools, and frameworks.
*   🚀 **Simplified Automation:** YAML-based configurations and automatic agent generation streamline the development process.

## What's New

*   **[2025-09-09]** Hosted a live sharing the design philosophy and basic usage of `Youtu-Agent`. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   **[2025-09-02]** [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **[2025-08-28]** Hosted a live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## Benchmark Performance

Youtu-Agent achieves strong performance on challenging benchmarks, demonstrating the effectiveness of its open-source approach.

*   **WebWalkerQA:**  Achieved 71.47% accuracy with DeepSeek-V3.1.
*   **GAIA:**  Achieved 72.8% pass@1 on the text-only validation subset.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples: Showcasing Agent Capabilities

Explore real-world applications of Youtu-Agent with these interactive examples:

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyzes a CSV file and generates an HTML report.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Renames and categorizes local files.
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/60193435-b89d-47d3-8153-5799d6ff2920" 
             poster="https://img.youtube.com/vi/r9we4m1cB6M/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e" 
             poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
  <tr >
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Wide Research</strong><br>Generates a comprehensive report, replicating Manus functionality.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Analyzes and compiles related literature.
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92" 
             poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d" 
             poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
</table>

> [!NOTE]
> Explore additional examples and detailed usage in the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/).

### 🤖 Automated Agent Configuration

Youtu-Agent simplifies agent creation with its automatic agent generation feature, reducing manual effort and accelerating development.

```bash
# Interactively define requirements and generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interact to clarify your requirements and generate the agent configuration.
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding:10px; vertical-align:top; width: 400px;">
      <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef" 
             poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="auto" 
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
</table>

> [!NOTE]
> Refer to the [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for details on automated agent generation.

## Core Concepts

*   **Agent:** LLM configured with prompts, tools, and an environment.
*   **Toolkit:**  Encapsulated set of tools the agent can use.
*   **Environment:**  The operating context (e.g., browser, shell).
*   **ContextManager:** Module for managing the agent's context window.
*   **Benchmark:**  Workflow for a specific dataset, including pre-processing, rollout, and judging logic.

For in-depth details on design and implementation, consult the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Get Started

Follow these steps to quickly run your first agent:

### Setup

#### Source Code Deployment

1.  **Prerequisites:** Python 3.12+ and [uv](https://github.com/astral-sh/uv) recommended for dependency management.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    ```
3.  **Install Dependencies:**
    ```bash
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Configure your API keys here
    ```

    *   Fill in your LLM API keys in the `.env` file (e.g., `UTU_LLM_API_KEY`).
    *   [Tencent Cloud](https://www.tencentcloud.com/) offers free DeepSeek API tokens for new users (Sep 1 – Oct 31, 2025); use them and update `.env` accordingly.

#### Docker Deployment

For a streamlined setup, refer to [`docker/README.md`](./docker/README.md).

### Quick Start

Use the `configs/agents/simple/base_search.yaml` configuration to run an agent with a search tool:

```bash
#  Set SERPER_API_KEY and JINA_API_KEY in .env for web search.
python scripts/cli_chat.py --stream --config simple/base_search
# To avoid search toolkit, run:
python scripts/cli_chat.py --stream --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Run Examples

Configure tool APIs (e.g., `SERPER_API_KEY`, `JINA_API_KEY`) in `.env` for examples requiring web search.
Here’s how to generate an SVG image on “DeepSeek V3.1 New Features”:

```bash
python examples/svg_generator/main.py
```

To visualize the agent's runtime status with the web UI:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

```bash
python examples/svg_generator/main_web.py
```
Access the project via the local link provided.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)
![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on standard datasets (e.g., WebWalkerQA):

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```
Results are stored and can be further analyzed. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📚 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**

## 🙏 Acknowledgements

This project is built upon the contributions of:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contribute

Join the community!  See our [**Contributing Guidelines**](./CONTRIBUTING.md) to contribute.

## 📚 Citation

```bibtex
@misc{youtu-agent-2025,
  title={Youtu-agent: A Simple yet Powerful Agent Framework},
  author={Tencent Youtu Lab},
  year={2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TencentCloudADP/youtu-agent}},
}
```

## ⭐ Star History

![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)