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

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

[English](./README.md) | [中文](./README_zh.md)

<div align="center">
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%"/>
</div>

## OxyGent: Build Intelligent Systems Faster with This Open-Source Framework

**OxyGent** is a cutting-edge Python framework designed to accelerate the development of production-ready intelligent systems. This framework empowers developers to create, deploy, and evolve sophisticated multi-agent systems with ease.  Find the original repository [here](https://github.com/jd-opensource/OxyGent).

### Key Features:

*   **Efficient Development:**
    *   Modular multi-agent framework using standardized components for rapid assembly.
    *   Supports hot-swapping and cross-scenario reuse via clean Python interfaces.

*   **Intelligent Collaboration:**
    *   Dynamic planning paradigms enable agents to decompose tasks and negotiate solutions.
    *   Agents adapt to changes and maintain full auditability of every decision.

*   **Elastic Architecture:**
    *   Supports diverse agent topologies, from simple ReAct to complex hybrid patterns.
    *   Automated dependency mapping and visual debugging tools for performance optimization.

*   **Continuous Evolution:**
    *   Built-in evaluation engines generate training data for continuous agent improvement.
    *   Maintains full transparency through knowledge feedback loops.

*   **Scalability:**
    *   Distributed scheduler facilitates linear cost growth with exponential gains.
    *   Handles domain-wide optimization and real-time decision-making at any scale.

*   **Competitive Performance:**
    *   OxyGent achieved 59.14 points on the GAIA benchmark (July 15, 2025).

### 1. Project Overview

OxyGent is an open-source framework that unifies tools, models, and agents into modular Oxy. It provides transparent, end-to-end pipelines, facilitating the seamless building, running, and evolution of multi-agent systems.

### 2. Framework Core Classes

<div align="center">
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" width="80%"/>
</div>

### 3. Feature Highlights

*   **For Developers:** Focus on business logic without needing to reinvent the wheel.
*   **For Enterprises:** Unifies AI systems, reducing communication overhead.
*   **For Users:** Provides a seamless experience through an intelligent agent ecosystem.

### 4. Quick Start

Follow these steps to quickly get started with OxyGent:

1.  **Create and Activate a Python Environment:**

    *   **Method 1: conda**

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

2.  **Install the Required Python Package:**

    *   **Method 1: conda**

        ```bash
        pip install oxygent
        ```

    *   **Method 2: uv**

        ```bash
        uv pip install oxygent
        ```

    *   **Method 3: Set Development Environment**

        ```bash
        git clone https://github.com/jd-opensource/OxyGent.git
        cd OxyGent
        pip install -r requirements.txt # or in uv
        brew install coreutils # maybe essential
        ```

3.  **(If Using MCP) Node.js Environment:**

    *   Download and install [Node.js](https://nodejs.org)

4.  **Write a Sample Python Script (demo.py):**

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

5.  **Set Environment Variables:**

    *   **Method 1: Declare in Terminal**

        ```bash
        export DEFAULT_LLM_API_KEY="your_api_key"
        export DEFAULT_LLM_BASE_URL="your_base_url"
        export DEFAULT_LLM_MODEL_NAME="your_model_name"
        ```

    *   **Method 2: Create a .env file**

        ```bash
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
        <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" width="80%"/>
    </div>

### 5. Contributing

We welcome contributions! You can contribute by:

1.  Reporting Issues (Bugs & Errors)
2.  Suggesting Enhancements
3.  Improving Documentation
    *   Fork the repository
    *   Add your view in the document
    *   Send your pull request
4.  Writing Code
    *   Fork the repository
    *   Create a new branch
    *   Add your feature or improvement
    *   Send your pull request

For detailed information and assistance, please consult our documentation: * **[Document](http://oxygent.jd.com/docs/)**

### 6. Community & Support

If you encounter issues, please submit reproducible steps and log snippets in the project's Issues area, or contact the OxyGent Core team via your internal Slack.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

### 7. About the Contributors

Thank you to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

### 8. License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!