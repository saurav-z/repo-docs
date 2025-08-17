# OxyGent: Build Production-Ready Intelligent Systems with Ease

**OxyGent is a powerful, open-source framework that empowers developers to rapidly build and deploy advanced, multi-agent AI systems.**

[View the Original Repository on GitHub](https://github.com/jd-opensource/OxyGent)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%">
  <br>
  <a href="http://oxygent.jd.com">Visit our website</a>
</div>

## Key Features

*   **🚀 Efficient Development:** Build, deploy, and evolve AI teams with unparalleled speed using modular Oxy components. Standardized interfaces enable rapid agent assembly, hot-swapping, and cross-scenario reuse.
*   **🤝 Intelligent Collaboration:**  Facilitate dynamic planning, task decomposition, and real-time adaptation through intelligent agents. Maintain full auditability of every decision.
*   **🕸️ Elastic Architecture:** Supports any agent topology from simple ReAct to complex hybrid planning. Automated dependency mapping and visual debugging optimize performance.
*   **🔁 Continuous Evolution:** Leverage built-in evaluation engines to generate training data and enable continuous improvement through knowledge feedback loops, all while maintaining full transparency.
*   **📈 Scalability:**  Scale effortlessly with a distributed scheduler, enabling linear cost growth while delivering exponential gains in collaborative intelligence. Handles domain-wide optimization and real-time decision making.

## Overview

OxyGent is an open-source framework designed to streamline the development of intelligent systems. It combines tools, models, and agents into a modular framework, enabling developers to create, run, and improve multi-agent systems efficiently.  OxyGent delivers a complete production lifecycle, including agent code in Python, one-command deployment, comprehensive decision monitoring, and automated evolution.

## Architecture

### Diagram

<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="Architecture Diagram" width="100%">

### Components

*   **📦 Repository:** Organizes agents, tools, LLMs, data, and system files in a unified structure.
*   **🛠 Production Framework:** Provides a complete production chain, including registration, building, running, evaluation, and evolution.
*   **🖥 Service Framework:** Offers a complete business system server with storage and monitoring support.
*   **⚙️ Engineering Base:** Provides rich external support, including integrated modules for databases and inference engines.

## Feature Highlights

*   **For Developers:**  Focus on business logic instead of repetitive infrastructure tasks.
*   **For Enterprises:**  Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

## Quick Start

1.  **Set up your environment:**

    *   **Using Conda:**

        ```bash
        conda create -n oxy_env python==3.10
        conda activate oxy_env
        pip install oxygent
        ```
    *   **Using uv:**

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv python install 3.10
        uv venv .venv --python 3.10
        source .venv/bin/activate
        uv pip install oxygent
        ```
2.  **Alternatively, for development:**

    *   Download **[Node.js](https://nodejs.org)**
    *   Install requirements:

        ```bash
        pip install -r requirements.txt  # or uv pip install -r requirements.txt
        brew install coreutils # maybe essential
        ```

3.  **Write a sample Python script:**

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

4.  **Configure your LLM settings:**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

    or

    ```bash
    # create a .env file
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

5.  **Run the example:**

    ```bash
    python demo.py
    ```

6.  **View the output:**

    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="Output Example" width="100%">

## Contributing

Contribute to OxyGent in several ways:

1.  Report Issues (Bugs & Errors)
2.  Suggest Enhancements
3.  Improve Documentation:
    *   Fork the repository
    *   Add your view in document
    *   Send your pull request
4.  Write Code:
    *   Fork the repository
    *   Create a new branch
    *   Add your feature or improvement
    *   Send your pull request

For development-related issues, please check out the [OxyGent Documentation](http://oxygent.jd.com/docs/).

## Community & Support

Submit reproducible steps and log snippets in the project's Issues area or contact the OxyGent Core team via your internal Slack.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## Contributors

Thanks to all the following [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

**OxyGent is provided by Oxygen JD.com.**

**Thanks for your contributions!**