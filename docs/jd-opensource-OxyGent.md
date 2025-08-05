# OxyGent: Build Production-Ready Intelligent Systems with Ease

OxyGent is an open-source Python framework designed to accelerate the development and deployment of multi-agent systems, making complex AI projects simpler and more efficient. **[Explore OxyGent on GitHub](https://github.com/jd-opensource/OxyGent)**.

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%"/>
</p>

## Key Features

*   **Efficient Development:**
    *   Modular multi-agent framework for building, deploying, and evolving AI teams efficiently.
    *   Standardized Oxy components for rapid agent assembly.
    *   Supports hot-swapping and cross-scenario reuse through clean Python interfaces.
*   **Intelligent Collaboration:**
    *   Dynamic planning paradigms for intelligent task decomposition and solution negotiation.
    *   Agents adapt to real-time changes and maintain full auditability.
*   **Elastic Architecture:**
    *   Supports various agent topologies, from simple ReAct to complex hybrid patterns.
    *   Automated dependency mapping and visual debugging tools for performance optimization.
*   **Continuous Evolution:**
    *   Built-in evaluation engines that auto-generate training data.
    *   Agents continuously improve through knowledge feedback loops with full transparency.
*   **Scalability:**
    *   Distributed scheduler enables linear cost growth with exponential gains in collaborative intelligence.
    *   Effortlessly handles domain-wide optimization and real-time decision-making at any scale.

## Why Choose OxyGent?

OxyGent empowers developers to focus on business logic rather than infrastructure. Enterprises can replace siloed AI systems with a unified framework, reducing communication overhead. Users will experience seamless teamwork within an intelligent agent ecosystem. The framework enables you to:

1.  **Code** agents in Python (without YAML headaches).
2.  **Deploy** with a single command (locally or in the cloud).
3.  **Monitor** every decision (full transparency).
4.  **Evolve** automatically (self-improving systems).

## Architecture

### Diagram

<p align="center">
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="OxyGent Architecture" width="100%"/>
</p>

### Description

*   **Repository:** Stores agents, tools, LLMs, data, and system files in a unified structure.
*   **Production Framework:** Complete production chain including registration, building, running, evaluation, and evolution.
*   **Service framework:** Complete business system server, providing comprehensive storage and monitoring support.
*   **Engineering base:** Rich external support, including integrated modules such as databases and inference engines.

## Quick Start

### Prerequisites

*   Python 3.10+
*   Ensure `pip` or `uv` and `node.js` are installed.

### Installation

**Using Conda:**

```bash
conda create -n oxy_env python==3.10
conda activate oxy_env
pip install oxygent
```

**Using UV:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install oxygent
```

### Example Code

```python
import os
from oxygent import MAS, Config, oxy, preset_tools

Config.set_agent_llm_model("default_llm")

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
        llm_params={"temperature": 0.01},
        semaphore=4,
    ),
    preset_tools.time_tools,
    oxy.ReActAgent(
        name="time_agent",
        desc="A tool that can query the time",
        tools=["time_tools"],
    ),
    preset_tools.file_tools,
    oxy.ReActAgent(
        name="file_agent",
        desc="A tool that can operate the file system",
        tools=["file_tools"],
    ),
    preset_tools.math_tools,
    oxy.ReActAgent(
        name="math_agent",
        desc="A tool that can perform mathematical calculations.",
        tools=["math_tools"],
    ),
    oxy.ReActAgent(
        is_master=True,
        name="master_agent",
        sub_agents=["time_agent", "file_agent", "math_agent"],
    ),
]

async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="What time is it now? Please save it into time.txt.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### LLM Configuration

Set your LLM API key, base URL, and model name:

```bash
export DEFAULT_LLM_API_KEY="your_api_key"
export DEFAULT_LLM_BASE_URL="your_base_url"
export DEFAULT_LLM_MODEL_NAME="your_model_name"
```

Alternatively, use a `.env` file:

```
DEFAULT_LLM_API_KEY="your_api_key"
DEFAULT_LLM_BASE_URL="your_base_url"
DEFAULT_LLM_MODEL_NAME="your_model_name"
```

### Run the Example

```bash
python demo.py
```

### Output

<p align="center">
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="Output Example" width="50%"/>
</p>

## Contributing

We welcome contributions!  You can contribute by:

*   Reporting issues (bugs & errors)
*   Suggesting enhancements
*   Improving documentation
*   Writing code

### Contribution Guidelines

1.  Fork the repository.
2.  Create a new branch.
3.  Add your feature or improvement.
4.  Send your pull request.

## Community & Support

For any issues, submit reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area.

Contact the OxyGent Core team via internal channels.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thanks to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!