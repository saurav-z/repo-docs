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

<!-- English](./README.md) | [中文](./README_zh.md) -->

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

<h1 align="center">OxyGent: Build Production-Ready Intelligent Systems with Ease</h1>
<p align="center">
  <a href="https://github.com/jd-opensource/OxyGent">
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%" alt="OxyGent Banner"/>
  </a>
</p>

OxyGent is a powerful open-source Python framework designed to streamline the development of sophisticated, production-ready intelligent systems.

**[Visit the original repo on GitHub](https://github.com/jd-opensource/OxyGent)**

## Key Features

*   ✅ **Efficient Development:** Build, deploy, and evolve AI teams with unprecedented efficiency using modular, reusable Oxy components.
*   🤝 **Intelligent Collaboration:** Enable dynamic planning paradigms where agents decompose tasks, negotiate, and adapt in real-time.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies, from simple to complex, with automated dependency mapping and visual debugging.
*   🔁 **Continuous Evolution:** Leverage built-in evaluation engines for automated training data generation, fostering continuous agent improvement.
*   📈 **Scalability:** Benefit from a distributed scheduler that delivers linear cost growth with exponential gains in collaborative intelligence.

## What is OxyGent?

OxyGent is an open-source framework that unifies tools, models, and agents into modular Oxy. Empowering developers with transparent, end-to-end pipelines, OxyGent makes building, running, and evolving multi-agent systems seamless and infinitely extensible.

## Project Highlights

*   **Achieves Competitive Performance:** The latest version of OxyGent scored 59.14 points on the GAIA benchmark, demonstrating its capabilities.
*   **Focus on Developer Experience:** OxyGent allows developers to concentrate on business logic without needing to reinvent the wheel.
*   **Enterprise-Ready:** Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **User-Centric:** Experience seamless teamwork from an intelligent agent ecosystem.

## Quickstart Guide

Follow these steps to get started with OxyGent:

### Step 1: Create and Activate a Python Environment
   *   **Method 1: Conda**
        ```bash
        conda create -n oxy_env python==3.10
        conda activate oxy_env
        ```
   *   **Method 2: uv**
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv python install 3.10 
        uv venv .venv --python 3.10
        source .venv/bin/activate
        ```

### Step 2: Install the OxyGent Package
   *   **Method 1: Conda**
        ```bash
        pip install oxygent
        ```
   *   **Method 2: uv**
        ```bash
        uv pip install oxygent
        ```
   *   **Method 3: Develop Environment**
        ```bash
        git clone https://github.com/jd-opensource/OxyGent.git
        cd OxyGent
        pip install -r requirements.txt # or in uv
        brew install coreutils # maybe essential
        ```

### Step 3: Install Node.js (if using MCP)
    *   Download and install [Node.js](https://nodejs.org)

### Step 4: Write a Sample Python Script (`demo.py`)
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
          await mas.start_web_service(
             first_query="What time is it now? Please save it into time.txt."
          )

    if __name__ == "__main__":
       import asyncio
       asyncio.run(main())
    ```

### Step 5: Set Environment Variables
   *   **Method 1: Terminal**
        ```bash
        export DEFAULT_LLM_API_KEY="your_api_key"
        export DEFAULT_LLM_BASE_URL="your_base_url"
        export DEFAULT_LLM_MODEL_NAME="your_model_name"
        ```
   *   **Method 2: .env file**
        ```
        DEFAULT_LLM_API_KEY="your_api_key"
        DEFAULT_LLM_BASE_URL="your_base_url"
        DEFAULT_LLM_MODEL_NAME="your_model_name"
        ```

### Step 6: Run the Example
   ```bash
   python demo.py
   ```

### Step 7: View the Output
*   See the example output in the original README.

## Core Classes

*   *Refer to the images in the original README for the framework's core classes.*

## Contribute

We welcome contributions of all kinds!

*   Report Issues (Bugs & Errors)
*   Suggest Enhancements
*   Improve Documentation
*   Write Code

For details, see the "Contributing" section in the original README.

## Community & Support

If you encounter any issues, please submit reproducible steps and log snippets in the project's [Issues area](https://github.com/jd-opensource/OxyGent/issues), or contact the OxyGent Core team directly.

## Contact Us

*   *Refer to the image in the original README for the team's contact details.*

## Contributors

Thanks to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

*   *Refer to the image in the original README for the contributor list.*

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!