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

<div align="center">
  <a href="https://github.com/jd-opensource/OxyGent">
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%" alt="OxyGent Banner"/>
  </a>
</div>

<h1 align="center">OxyGent: Build Advanced Intelligent Systems with Ease</h1>

OxyGent is a cutting-edge, open-source Python framework designed to accelerate the development and deployment of production-ready intelligent systems. [Explore the OxyGent repository](https://github.com/jd-opensource/OxyGent).

## Key Features

*   **Efficient Development:** Build, deploy, and evolve AI teams quickly with a modular multi-agent framework using clean Python interfaces.
*   **Intelligent Collaboration:** Leverage dynamic planning paradigms for intelligent task decomposition, solution negotiation, and real-time adaptation.
*   **Elastic Architecture:** Support any agent topology, from simple ReAct to complex hybrid planning patterns, optimizing performance across distributed systems.
*   **Continuous Evolution:** Enhance agents through knowledge feedback loops and built-in evaluation engines that auto-generate training data.
*   **Scalability:** Scale seamlessly with a distributed scheduler, delivering exponential gains in collaborative intelligence.

## Project Overview

OxyGent is an open-source framework designed to streamline the creation, execution, and evolution of multi-agent systems. It combines tools, models, and agents into modular "Oxy" components, enabling developers to build end-to-end pipelines with transparency and extensibility.

## Framework Core Classes

[Insert image of core classes:  `https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png`]

## Feature Highlights

*   **For Developers:** Focus on business logic without reinventing the wheel.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

## Quick Start Guide

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

### Step 2: Install the Required Python Package

*   **Method 1: Conda**

    ```bash
    pip install oxygent
    ```
*   **Method 2: uv**

    ```bash
    uv pip install oxygent
    ```
*   **Method 3: Set Develop Environment**

    ```bash
    git clone https://github.com/jd-opensource/OxyGent.git
    cd OxyGent
    pip install -r requirements.txt # or in uv
    brew install coreutils # maybe essential
    ```

### Step 3: Node.js Environment (if using MCP)

*   Download and install **[Node.js](https://nodejs.org)**.

### Step 4: Write a Sample Python Script

*   Create `demo.py` with the following content:

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

*   **Method 1: Declare in Terminal**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```
*   **Method 2: Create a .env File**

    ```
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

### Step 6: Run the Example

*   Start the multi-agent system:

    ```bash
    python demo.py
    ```

### Step 7: View the Output

[Insert image of output:  `https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png`]

## Contributing

We welcome contributions to OxyGent!  You can contribute by:

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

For development-related questions, please refer to the document: * **[Document](http://oxygent.jd.com/docs/)**

## Community & Support

If you have questions or encounter issues, please submit reproducible steps and log snippets in the project's Issues area or contact the OxyGent Core team.

[Insert image for contact details:  `https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216`]

## About the Contributors

Thanks to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!