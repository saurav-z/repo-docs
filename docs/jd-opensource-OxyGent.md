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

**OxyGent is a cutting-edge Python framework designed to streamline the development and deployment of advanced, multi-agent AI systems.**  [View the original repository on GitHub](https://github.com/jd-opensource/OxyGent).

## Key Features

*   🚀 **Efficient Development:** Build, deploy, and evolve AI teams with unprecedented efficiency using modular components.
    *   Standardized components enable rapid agent assembly.
    *   Supports hot-swapping and cross-scenario reuse.
    *   Clean Python interfaces without messy configurations.

*   🤝 **Intelligent Collaboration:** Supercharge collaboration with dynamic planning, enabling agents to:
    *   Intelligently decompose tasks and negotiate solutions.
    *   Adapt to changes in real-time.
    *   Maintain full auditability of every decision.

*   🕸️ **Elastic Architecture:** Supports any agent topology and offers:
    *   Automated dependency mapping.
    *   Visual debugging tools for performance optimization.

*   🔁 **Continuous Evolution:** Built-in evaluation engines that:
    *   Auto-generate training data.
    *   Continuously improve agents through knowledge feedback loops.
    *   Maintain full transparency.

*   📈 **Scalability:** Distributed scheduler enables linear cost growth with exponential gains in collaborative intelligence.
    *   Effortlessly handles domain-wide optimization.
    *   Real-time decision-making at any scale.

## Performance

OxyGent achieved 59.14 points in the [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) benchmark (July 15, 2025), close to the current top opensource system OWL (60.8 points).

![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png)

## Architecture

### Diagram
![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png)

### Description

*   📦 **Repository**: Unified structure for storing agents, tools, LLMs, data, and system files.
*   🛠 **Production Framework**: Complete production chain including registration, building, running, evaluation, and evolution.
*   🖥 **Service Framework**: Complete business system server, providing storage and monitoring support.
*   ⚙️ **Engineering Base**: Extensive external support, including integrated modules such as databases and inference engines.

## Benefits

*   **For Developers:** Focus on business logic, not infrastructure.
*   **For Enterprises:** Replace siloed AI systems with a unified framework.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

We provide the complete lifecycle:
1.  **Code** agents in Python.
2.  **Deploy** with one command.
3.  **Monitor** every decision.
4.  **Evolve** automatically.

## Quickstart

1.  **Create and activate a Python environment:**

    ```bash
       conda create -n oxy_env python==3.10
       conda activate oxy_env
    ```
    or
    ```bash
       curl -LsSf https://astral.sh/uv/install.sh | sh
       uv python install 3.10
       uv venv .venv --python 3.10
       source .venv/bin/activate
    ```

2.  **Install OxyGent:**

    ```bash
       pip install oxygent
    ```
    or
    ```bash
       uv pip install oxygent
    ```

3.  **Or set develop environment:**
    *   Download **[Node.js](https://nodejs.org)**
    *   Download our requirements:
        ```bash
            pip install -r requirements.txt
            brew install coreutils # maybe essential
        ```

4.  **Write a sample Python script** (e.g., `demo.py`):

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

    Or create a `.env` file:

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

    ![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)

## Contributing

We welcome contributions!  You can contribute by:

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

For development-related questions, please consult our [documentation](http://oxygent.jd.com/docs/).

## Community & Support

Submit reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area, or contact the OxyGent Core team via your internal Slack.

Contact us:

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thanks to all the following [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License](./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!