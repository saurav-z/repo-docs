# OxyGent: Build Production-Ready Intelligent Systems with Ease

**OxyGent empowers developers to build, deploy, and evolve sophisticated multi-agent systems, enabling rapid innovation in AI.** ([View on GitHub](https://github.com/jd-opensource/OxyGent))

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%">
</p>

## Key Features of OxyGent

*   🚀 **Efficient Development:** Rapidly build, deploy, and evolve AI teams using modular Oxy components. Supports hot-swapping, cross-scenario reuse, and clean Python interfaces.
*   🤝 **Intelligent Collaboration:** Enables dynamic planning, task decomposition, negotiation, and real-time adaptation for intelligent agents. Full auditability of all decisions.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies and provides automated dependency mapping and visual debugging tools for optimal performance.
*   🔁 **Continuous Evolution:** Built-in evaluation engines generate training data, allowing agents to continuously improve through knowledge feedback loops.
*   📈 **Scalability:**  A distributed scheduler enables linear cost growth and exponential gains in collaborative intelligence, handling domain-wide optimization at any scale.

*As of July 15, 2025, OxyGent achieved 59.14 points on the GAIA benchmark, rivaling top open-source systems.*
<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png" width="50%">
</p>

## 1. Overview

**OxyGent** is an open-source Python framework designed to streamline the development of intelligent systems. It unifies tools, models, and agents into modular "Oxy" components. With transparent end-to-end pipelines, OxyGent simplifies the creation, execution, and evolution of multi-agent systems.

## 2. Software Architecture

### 2.1 Diagram

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" width="80%">
</p>

### 2.2 Architecture Description

*   📦 **Repository**: Organizes agents, tools, LLMs, data, and system files in a unified structure.
*   🛠 **Production Framework**: Provides a comprehensive production pipeline including registration, building, running, evaluation, and evolution capabilities.
*   🖥 **Service Framework**: Implements a full business system server, with storage and monitoring support.
*   ⚙️ **Engineering Base**: Integrates with databases, inference engines, and other external resources.

## 3. Feature Highlights

*   **For Developers:** Focus on core business logic without reinventing the wheel.
*   **For Enterprises:** Consolidate siloed AI systems into a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

**Key Lifecycle Components:**

1.  **Code** agents in Python.
2.  **Deploy** with a single command.
3.  **Monitor** every decision for full transparency.
4.  **Evolve** automatically with self-improving systems.

## 4. Quick Start

1.  **Set up your environment (conda):**

    ```bash
    conda create -n oxy_env python==3.10
    conda activate oxy_env
    ```
    or **(uv)**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.10
    uv venv .venv --python 3.10
    source .venv/bin/activate
    ```

2.  **Install OxyGent (conda):**

    ```bash
    pip install oxygent
    ```
    or **(uv)**
    ```bash
    uv pip install oxygent
    ```

3.  **Alternatively, for development:**

    *   Install [Node.js](https://nodejs.org).
    *   Install dependencies:

    ```bash
       pip install -r requirements.txt # or in uv
       brew install coreutils # maybe essential
    ```

4.  **Create a sample Python script (demo.py):**

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

    ```
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

6.  **Run the example:**

    ```bash
    python demo.py
    ```

7.  **View the output:**

    <p align="center">
      <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" width="50%">
    </p>

## 5. Contributing

Contribute to OxyGent!

1.  Report Issues (Bugs & Errors)
2.  Suggest Enhancements
3.  Improve Documentation
    *   Fork the repository
    *   Add your view in document
    *   Send your pull request
4.  Write Code
    *   Fork the repository
    *   Create a new branch
    *   Add your feature or improvement
    *   Send your pull request

For development issues, check our documentation: [Document](http://oxygent.jd.com/docs/)

## 6. Community & Support

Submit issues, or contact the OxyGent Core team directly via internal Slack.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## 7. Contributors

Thanks to all the contributors:
<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## 8. License

[Apache License]( ./LICENSE.md)

**OxyGent is provided by Oxygen JD.com.**
**Thanks for your Contributions!**