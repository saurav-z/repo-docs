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

# OxyGent: Build Production-Ready Intelligent Systems with Python

**OxyGent is a cutting-edge Python framework that revolutionizes the development of multi-agent systems, enabling rapid prototyping and scalable deployment.**  Find the original repository [here](https://github.com/jd-opensource/OxyGent).

## Key Features

*   **Rapid Development:** Build, deploy, and evolve AI teams with unprecedented efficiency using modular, reusable components.
*   **Intelligent Collaboration:** Enable agents to dynamically plan, negotiate, and adapt to real-time changes for robust problem-solving.
*   **Elastic Architecture:** Support a wide range of agent topologies and optimize performance with automated dependency mapping and visual debugging tools.
*   **Continuous Evolution:** Leverage built-in evaluation engines to automatically generate training data, driving continuous agent improvement and full transparency.
*   **Scalability:**  Scale your collaborative intelligence linearly, handling domain-wide optimization and real-time decision-making at any scale.
*   **Simplified Workflow:** Code agents in Python, deploy with a single command, monitor every decision, and evolve automatically.
*   **Production-Ready:** Designed for building AI infrastructure that works seamlessly in production environments.

## Project Overview

OxyGent is an open-source framework that unifies tools, models, and agents into modular Oxy components, enabling the creation of end-to-end, transparent AI pipelines. This makes building, running, and evolving multi-agent systems seamless and extensible.

## Software Architecture

### 3.1 Diagram
![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png)
<!-- Insert architecture diagram here -->
### 3.2 Architecture Description
*   📦 **Repository**: Stores agents, tools, LLMs, data, and system files in a unified structure.
*   🛠 **Production Framework**: A complete production chain that includes registration, building, running, evaluation, and evolution.
*   🖥 **Service framework**: complete business system server, providing complete storage and monitoring support
*   ⚙️ **Engineering base**: Rich external support, including integrated modules such as databases and inference engines

## Feature Highlights

*   **For Developers:** Focus on business logic without reinventing the wheel.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

## Quick Start

1.  **Set up your environment:**
    *   Create and activate a Python environment using `conda`:
        ```bash
        conda create -n oxy_env python==3.10
        conda activate oxy_env
        ```
        or using `uv`:
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv python install 3.10 
        uv venv .venv --python 3.10
        source .venv/bin/activate
        ```
    *   Install the required Python package (using `conda`):
        ```bash
        pip install oxygent
        ```
        or (using `uv`):
        ```bash
        uv pip install oxygent
        ```

2.  **Alternatively, set up a development environment:**
    *   Download **[Node.js](https://nodejs.org)**
    *   Download our requirements:
        ```bash
        pip install -r requirements.txt # or in uv
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
    or create a `.env` file:
    ```
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

5.  **Run the example:**
    ```bash
    python demo.py
    ```

6.  **View the output:**
    ![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)

## Contributing

Contribute to OxyGent by:

1.  Reporting Issues (Bugs & Errors)
2.  Suggesting Enhancements
3.  Improving Documentation:
    *   Fork the repository
    *   Add your view in document
    *   Send your pull request
4.  Writing Code:
    *   Fork the repository
    *   Create a new branch
    *   Add your feature or improvement
    *   Send your pull request

For further details on development, refer to our documentation: * **[Document](http://oxygent.jd.com/docs/)**

## Community & Support

If you encounter issues, submit reproducible steps and log snippets in the project's Issues area or contact the OxyGent Core team.

Contact us:

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thank you to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License](./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!