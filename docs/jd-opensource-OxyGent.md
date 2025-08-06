# OxyGent: Build Production-Ready Intelligent Systems with Ease

OxyGent is an open-source Python framework designed to streamline the development of multi-agent systems for real-world applications. [Explore the original repository on GitHub](https://github.com/jd-opensource/OxyGent) for more details and to contribute.

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%"/>
</p>

## Key Features

*   🚀 **Efficient Development:** Build, deploy, and evolve AI teams with unparalleled efficiency using modular components.
    *   Standardized components for rapid agent assembly.
    *   Hot-swapping and cross-scenario reuse.
    *   Clean Python interfaces.
*   🤝 **Intelligent Collaboration:** Facilitate dynamic planning, task decomposition, and solution negotiation among agents.
    *   Agents adapt to changing environments in real-time.
    *   Full auditability of every decision.
*   🕸️ **Elastic Architecture:**  Supports various agent topologies and optimizes performance across distributed systems.
    *   Automated dependency mapping.
    *   Visual debugging tools.
*   🔁 **Continuous Evolution:** Improve agents through built-in evaluation engines and knowledge feedback loops.
    *   Auto-generated training data.
    *   Full transparency throughout the learning process.
*   📈 **Scalability:**  Leverage a distributed scheduler for linear cost growth and exponential gains in collaborative intelligence.
    *   Domain-wide optimization and real-time decision making at any scale.

## Project Overview

OxyGent simplifies building, running, and evolving multi-agent systems by unifying tools, models, and agents into modular "Oxy" components.  It provides developers with transparent, end-to-end pipelines for creating production-ready intelligent systems.

## Software Architecture

### Architecture Diagram
*(Insert the architecture diagram here as described in the original README)*

### Architecture Description
*   📦 **Repository:** Organizes agents, tools, LLMs, data, and system files in a unified structure.
*   🛠️ **Production Framework:** Provides a complete production chain, including registration, building, running, evaluation, and evolution.
*   🖥️ **Service Framework:** Offers a comprehensive business system server, with storage and monitoring support.
*   ⚙️ **Engineering Base:** Includes rich external support and integrated modules like databases and inference engines.

## Feature Highlights

*   **For Developers:** Focus on business logic without the complexities of infrastructure.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork within an intelligent agent ecosystem.

**OxyGent offers a complete lifecycle:**

1.  **Code** agents in Python.
2.  **Deploy** with a single command (local or cloud).
3.  **Monitor** every decision with full transparency.
4.  **Evolve** automatically through continuous learning.

## Quick Start

1.  **Set up your environment:**

    *   Using Conda:
        ```bash
        conda create -n oxy_env python==3.10
        conda activate oxy_env
        ```
    *   Using `uv`:
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv python install 3.10
        uv venv .venv --python 3.10
        source .venv/bin/activate
        ```

2.  **Install OxyGent:**

    *   Using Conda:
        ```bash
        pip install oxygent
        ```
    *   Using `uv`:
        ```bash
        uv pip install oxygent
        ```

3.  **Alternatively, set up for development:**

    *   Download **[Node.js](https://nodejs.org)**
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt  # or use uv
        brew install coreutils  # may be essential
        ```

4.  **Create and run a sample script:**

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

5.  **Configure your LLM settings:**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```
    or create a `.env` file:
    ```bash
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

6.  **Run the example:**

    ```bash
    python demo.py
    ```

7.  **View the output:**

    *(Insert the output image here as described in the original README)*

## Contributing

Contribute to OxyGent in the following ways:

1.  Report issues (bugs and errors)
2.  Suggest Enhancements
3.  Improve Documentation
    *   Fork the repository
    *   Add your contribution
    *   Submit a pull request
4.  Write Code
    *   Fork the repository
    *   Create a new branch
    *   Add your feature or improvement
    *   Submit a pull request

All contributions are welcome!  For development-related questions, refer to the [documentation](http://oxygent.jd.com/docs/).

## Community & Support

For any issues, submit steps and log snippets in the project's Issues area, or contact the OxyGent Core team.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thanks to all [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### Provided by Oxygen JD.com
#### Thanks for Your Contributions!