<!-- Copyright 2022 JD Co.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

[English](./README.md) | [中文](./README_zh.md)

<p align="center">
  <a href="https://github.com/jd-opensource/OxyGent/pulls">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome">
  </a>
  <a href="https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="license"/>
  </a>
  <a href="https://pypi.org/project/oxygent/">
    <img src="https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white" alt="pip"/>
  </a>

<html>
    <h2 align="center">
      <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="1256"/>
    </h2>
    <h3 align="center">
      An advanced Python framework that empowers developers to quickly build production-ready intelligent systems.
    </h3>
    <h3 align="center">
      Visit our website:
      <a href="http://oxygent.jd.com">OxyGent</a>
    </h3>
</html>

# OxyGent: Build Intelligent Systems with Ease

**OxyGent is an open-source Python framework that revolutionizes the development of multi-agent systems, enabling efficient creation, deployment, and evolution of production-ready AI solutions.**

[View the original repository on GitHub](https://github.com/jd-opensource/OxyGent)

## Key Features

*   ✅ **Modular and Extensible:** Build AI teams efficiently with modular components that snap together like LEGO bricks, supporting hot-swapping and cross-scenario reuse.
*   🧠 **Intelligent Collaboration:** Leverage dynamic planning paradigms for agents to intelligently decompose tasks, negotiate solutions, and adapt to real-time changes.
*   ⚙️ **Elastic Architecture:** Supports any agent topology with automated dependency mapping and visual debugging tools for optimal performance.
*   🔄 **Continuous Evolution:** Utilize built-in evaluation engines for automatic training data generation, enabling continuous agent improvement and full transparency.
*   🚀 **Scalability:** Benefit from a distributed scheduler that enables linear cost growth while delivering exponential gains in collaborative intelligence.

The latest version of OxyGent (July 15, 2025) scored 59.14 points on the [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) benchmark, with the current top open-source system OWL achieving 60.8 points.

![OxyGent Benchmark Score](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png)

## Software Architecture

### Architecture Diagram

![OxyGent Architecture](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png)

### Architecture Components

*   📦 **Repository:** Unified storage for agents, tools, LLMs, data, and system files.
*   🛠 **Production Framework:** Complete production chain including registration, building, running, evaluation, and evolution.
*   🖥 **Service Framework:** Comprehensive business system server with full storage and monitoring support.
*   ⚙️ **Engineering Base:** Rich external support, including integrated modules such as databases and inference engines.

## Feature Highlights

*   **For Developers:** Focus on business logic without reinventing the wheel.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

**Complete Lifecycle:**

1.  **Code** agents in Python (no YAML hell)
2.  **Deploy** with one command (local or cloud)
3.  **Monitor** every decision (full transparency)
4.  **Evolve** automatically (self-improving systems)

## Quick Start

1.  **Set up your environment:**

    *   **Conda:**

        ```bash
        conda create -n oxy_env python==3.10
        conda activate oxy_env
        ```

    *   **UV:**

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv python install 3.10
        uv venv .venv --python 3.10
        source .venv/bin/activate
        ```

2.  **Install OxyGent:**

    *   **Conda:**

        ```bash
        pip install oxygent
        ```

    *   **UV:**

        ```bash
        uv pip install oxygent
        ```

3.  **Alternatively, Set up a development environment**

    *   Download **[Node.js](https://nodejs.org)**
    *   Download our requirements:

        ```bash
        pip install -r requirements.txt # or in uv
        brew install coreutils # maybe essential
        ```

4.  **Write a Sample Script:**

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

    ![OxyGent Output](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)

## Contributing

We welcome contributions! You can contribute by:

1.  Reporting Issues (Bugs & Errors)
2.  Suggesting Enhancements
3.  Improving Documentation
    *   Fork the repository
    *   Add your view in document
    *   Send your pull request
4.  Writing Code
    *   Fork the repository
    *   Create a new branch
    *   Add your feature or improvement
    *   Send your pull request

For any development issues, please refer to our documentation: **[Document](http://oxygent.jd.com/docs/)**

## Community & Support

For any issues, please submit reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area, or contact the OxyGent Core team via your internal Slack.

Welcome to contact us:

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thank you to all the contributors!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License](./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!