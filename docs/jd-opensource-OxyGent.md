# OxyGent: Build Production-Ready Intelligent Systems with Ease

**OxyGent is an open-source Python framework that empowers developers to build, deploy, and evolve advanced multi-agent systems for production.** ([View on GitHub](https://github.com/jd-opensource/OxyGent))

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%">
  <p><strong>OxyGent: The future of intelligent systems, built today.</strong></p>
  <p><a href="http://oxygent.jd.com">Visit our website</a></p>
</div>

## Key Features

*   **Modular Design**: Build AI teams with unparalleled efficiency using standardized, reusable components.
*   **Intelligent Collaboration**: Facilitates dynamic task decomposition, negotiation, and real-time adaptation for intelligent agents.
*   **Elastic Architecture**: Supports diverse agent topologies, from simple ReAct to complex hybrid planning patterns.
*   **Continuous Evolution**: Leverages built-in evaluation engines to auto-generate training data, ensuring continuous improvement.
*   **Scalability**: Offers a distributed scheduler for linear cost growth and exponential gains in collaborative intelligence.

## Project Overview

OxyGent is an open-source framework designed to unify tools, models, and agents into modular Oxy components. It provides developers with transparent, end-to-end pipelines for building, running, and evolving multi-agent systems seamlessly and with infinite extensibility.

## Software Architecture

### Diagram

<div align="center">
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" width="70%">
</div>

### Architecture Description

*   **Repository**: Stores agents, tools, LLMs, data, and system files in a unified structure.
*   **Production Framework**: A complete production chain that includes registration, building, running, evaluation, and evolution.
*   **Service Framework**: Comprehensive business system server, offering complete storage and monitoring support.
*   **Engineering Base**: Rich external support, including integrated modules such as databases and inference engines.

## Feature Highlights

*   **For Developers**: Focus on business logic without reinventing the wheel.
*   **For Enterprises**: Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **For Users**: Experience seamless teamwork from an intelligent agent ecosystem.

We've engineered the complete lifecycle:

1️⃣ **Code** agents in Python (no YAML hell)

2️⃣ **Deploy** with one command (local or cloud)

3️⃣ **Monitor** every decision (full transparency)

4️⃣ **Evolve** automatically (self-improving systems)

## Quick Start

1.  **Set Up Your Environment:**

    *   **Using Conda:**

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

    *   **Using Conda:**

        ```bash
        pip install oxygent
        ```

    *   **Using uv:**

        ```bash
        uv pip install oxygent
        ```

3.  **Alternatively, Set Up a Development Environment:**

    *   Download **[Node.js](https://nodejs.org)**
    *   Download the dependencies:
        ```bash
        pip install -r requirements.txt # or in uv
        brew install coreutils # maybe essential
        ```

4.  **Write a Sample Python Script:**

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

5.  **Configure Your LLM Settings:**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
    export DEFAULT_LLM_MODEL_NAME="your_model_name"  
    ```
    ```bash
    # create a .env file
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

6.  **Run the Example:**

    ```bash
    python demo.py
    ```

7.  **View the Output:**

    <div align="center">
        <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" width="70%">
    </div>

## Contributing

We welcome contributions! Here's how you can get involved:

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

For development-related questions, please see our documentation: **[Document](http://oxygent.jd.com/docs/)**.

## Community & Support

If you have any issues or need assistance, please submit reproducible steps and log snippets in the project's Issues area, or contact the OxyGent Core team directly.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thank you to all the contributors!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!