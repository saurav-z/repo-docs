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

<h1 align="center">OxyGent: Build Intelligent Systems with a Powerful Python Framework</h1>
<p align="center">
  <a href="https://github.com/jd-opensource/OxyGent">
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" alt="OxyGent Banner" width="100%">
  </a>
</p>

<p align="center">OxyGent is a cutting-edge Python framework that enables developers to rapidly build and deploy sophisticated, production-ready intelligent systems.</p>

## Key Features

*   🚀 **Modular & Efficient Development:** Build, deploy, and evolve AI teams with unprecedented speed using modular components for rapid agent assembly, hot-swapping, and cross-scenario reuse.
*   🤝 **Intelligent Collaboration:**  Facilitate dynamic planning, enabling agents to decompose tasks, negotiate solutions, and adapt in real-time, ensuring full auditability of every decision.
*   🕸️ **Elastic Architecture:** Support any agent topology, from simple ReAct to complex hybrid planning patterns, with automated dependency mapping and visual debugging tools for optimized performance.
*   🔁 **Continuous Evolution:** Improve agents through knowledge feedback loops using built-in evaluation engines that auto-generate training data, enhancing transparency and learning.
*   📈 **Scalability:** Leverage a distributed scheduler for linear cost growth and exponential gains in collaborative intelligence, handling domain-wide optimization and real-time decision-making at any scale.

## Project Overview

**OxyGent** is an open-source framework designed to unify tools, models, and agents into modular Oxy.  It provides developers with end-to-end pipelines, simplifying the building, running, and evolving of multi-agent systems, making them seamless and infinitely extensible.

## Framework Core Classes
[Include the image from the original README here:  ![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png)]

## Feature Highlights

*   **For Developers:** Focus on business logic without the need to reinvent the wheel.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, reducing communication overhead and increasing efficiency.
*   **For Users:** Experience seamless teamwork within an intelligent agent ecosystem.

## Quick Start Guide

Get up and running with OxyGent in a few simple steps:

### 1.  Set up your Python Environment

Choose your preferred method:

*   **Conda:**
    ```bash
    conda create -n oxy_env python==3.10
    conda activate oxy_env
    ```
*   **uv:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.10
    uv venv .venv --python 3.10
    source .venv/bin/activate
    ```

### 2. Install OxyGent

Choose your preferred method:

*   **Conda:**
    ```bash
    pip install oxygent
    ```
*   **uv:**
    ```bash
    uv pip install oxygent
    ```
*   **Develop Environment:**
    ```bash
    git clone https://github.com/jd-opensource/OxyGent.git
    cd OxyGent
    pip install -r requirements.txt # or in uv
    brew install coreutils # maybe essential
    ```

### 3. Node.js Environment (if using MCP)

Download and install **[Node.js](https://nodejs.org)**.

### 4. Write a Sample Python Script (demo.py)

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

### 5. Set Environment Variables

Choose your preferred method:

*   **Terminal:**
    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```
*   **.env file:**
    ```
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

### 6. Run the Example

```bash
python demo.py
```

### 7. View the Output

[Include the image from the original README here: ![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)]

## Contributing

Contribute to OxyGent through:

1.  Reporting Issues (Bugs & Errors)
2.  Suggesting Enhancements
3.  Improving Documentation (Fork, add your view, and submit a Pull Request)
4.  Writing Code (Fork, create a branch, add features, and submit a Pull Request)

We welcome all contributions! 🎉

For development-related issues, check our: * **[Document](http://oxygent.jd.com/docs/)**

## Community & Support

If you need help, submit reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area, or contact the OxyGent Core team via your internal Slack.

## Contact Us

[Include the contact image from the original README here: <div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>]

## About the Contributors

Thanks to all the following contributors:
<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!

For more details, visit the [OxyGent GitHub repository](https://github.com/jd-opensource/OxyGent).