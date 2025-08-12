# OxyGent: Build Production-Ready Intelligent Systems with Ease

**OxyGent is a cutting-edge Python framework empowering developers to build, deploy, and evolve sophisticated multi-agent systems efficiently.**

[View the original repository on GitHub](https://github.com/jd-opensource/OxyGent)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%">
  <p>
    <a href="http://oxygent.jd.com">Visit our website</a>
  </p>
</div>

## Key Features

*   🚀 **Efficient Development:** Build, deploy, and evolve AI teams with unparalleled efficiency through modular, reusable components.
*   🤝 **Intelligent Collaboration:** Leverage dynamic planning paradigms for intelligent task decomposition, negotiation, and real-time adaptation.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies and optimizes performance across distributed systems.
*   🔁 **Continuous Evolution:** Built-in evaluation engines provide feedback loops to continuously improve agent performance.
*   📈 **Scalability:** Designed for linear cost growth and exponential gains in collaborative intelligence.

## 1. Project Overview

OxyGent is an open-source framework that unifies tools, models, and agents into modular "Oxy" components. This innovative approach empowers developers to create transparent, end-to-end pipelines, streamlining the process of building, running, and evolving multi-agent systems.

## 2. Software Architecture

### 2.1 Diagram

<!-- Insert architecture diagram here -->
<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="OxyGent Architecture">

### 2.2 Architecture Description

*   📦 **Repository:**  Centralized storage for agents, tools, LLMs, data, and system files.
*   🛠️ **Production Framework:** Comprehensive production chain for registration, building, running, evaluation, and evolution.
*   🖥️ **Service Framework:** Complete business system server with storage and monitoring support.
*   ⚙️ **Engineering Base:** Rich external support, including database integration and inference engines.

## 3. Feature Highlights

*   **For Developers:** Focus on business logic without the hassle of reinventing the wheel.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, reducing communication overhead and costs.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

OxyGent simplifies the entire lifecycle:

1.  **Code** agents in Python.
2.  **Deploy** with a single command.
3.  **Monitor** every decision.
4.  **Evolve** automatically with continuous feedback loops.

## 4. Quick Start

1.  **Set up your environment:**

    *   **Using conda:**

        ```bash
        conda create -n oxy_env python==3.10
        conda activate oxy_env
        ```

    *   **Using uv:**

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv python install 3.10
        uv venv .venv --python 3.10
        source .venv/bin/activate
        ```

2.  **Install OxyGent:**

    *   **Using conda:**

        ```bash
        pip install oxygent
        ```

    *   **Using uv:**

        ```bash
        uv pip install oxygent
        ```

3.  **Alternatively, set up a development environment:**

    *   Download **[Node.js](https://nodejs.org)**
    *   Install requirements:

        ```bash
        pip install -r requirements.txt  # or in uv: uv pip install -r requirements.txt
        brew install coreutils # may be essential
        ```

4.  **Create a sample Python script:**

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

5.  **Configure your LLM settings (e.g., using environment variables):**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

    or  use a `.env` file:

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

    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="Output">

## 5. Contributing

We welcome contributions of all kinds!

*   Report issues (bugs and errors).
*   Suggest enhancements.
*   Improve documentation:
    1.  Fork the repository.
    2.  Add your view to the documentation.
    3.  Submit a pull request.
*   Write code:
    1.  Fork the repository.
    2.  Create a new branch.
    3.  Add your feature or improvement.
    4.  Submit a pull request.

For development issues, please check the document: **[Document](http://oxygent.jd.com/docs/)**

## 6. Community & Support

For assistance, please submit reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area or contact the OxyGent Core team via internal Slack.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## 7. About the Contributors

Thank you to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## 8. License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!