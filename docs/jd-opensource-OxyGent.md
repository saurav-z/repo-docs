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

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%" alt="OxyGent Banner">
</div>

<h2 align="center">OxyGent: Build Intelligent Systems Faster with a Powerful Python Framework</h2>

<div align="center">
  <a href="http://oxygent.jd.com">OxyGent Website</a>
</div>

**OxyGent** is an open-source Python framework revolutionizing the development of intelligent systems, enabling developers to build production-ready multi-agent systems with unprecedented efficiency.

## Key Features

*   **Efficient Development:** Build, deploy, and evolve AI teams quickly using modular Oxy components.
*   **Intelligent Collaboration:** Facilitates dynamic planning where agents decompose tasks, negotiate solutions, and adapt in real-time.
*   **Elastic Architecture:** Supports diverse agent topologies and optimized performance across distributed systems.
*   **Continuous Evolution:** Agents improve through knowledge feedback loops, with built-in evaluation engines.
*   **Scalability:** Enables linear cost growth and exponential gains in collaborative intelligence.

## 1. Project Overview

**OxyGent** provides a unified platform for creating and managing multi-agent systems. It combines tools, models, and agents into modular "Oxy" components. This allows for the rapid construction of AI solutions and supports continuous iteration and improvement.

## 2. Core Features in Detail

*   **Modularity:** Build agents from reusable components like LEGO bricks.
*   **Collaboration:** Implement dynamic planning for tasks.
*   **Flexibility:** Handles different agent architectures.
*   **Adaptability:** Allows agents to learn and improve.
*   **Scalability:** Optimized for large-scale deployments.

*As of July 15, 2025, OxyGent achieved 59.14 points on the GAIA leaderboard, close to OWL's 60.8 points.*

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png" alt="GAIA Score">
</div>

## 3. Framework Structure

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="Framework Structure">
</div>

## 4. Feature Highlights

*   **Developers:** Focus on business logic, not boilerplate.
*   **Enterprises:** Unify siloed AI systems.
*   **Users:** Experience seamless teamwork from an intelligent agent ecosystem.

## 5. Quick Start Guide

Get up and running with OxyGent in a few simple steps:

### Step 1: Set up a Python Environment

Choose either `conda` or `uv`.

**Conda:**

```bash
conda create -n oxy_env python==3.10
conda activate oxy_env
```

**uv:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10 
uv venv .venv --python 3.10
source .venv/bin/activate
```

### Step 2: Install OxyGent

Choose your preferred installation method:

**Conda:**

```bash
pip install oxygent
```

**uv:**

```bash
uv pip install oxygent
```

**Development Environment:**

```bash
git clone https://github.com/jd-opensource/OxyGent.git
cd OxyGent
pip install -r requirements.txt # or in uv
brew install coreutils # maybe essential
```

### Step 3: (If Using MCP) Install Node.js

Download and install **[Node.js](https://nodejs.org)**

### Step 4: Write a Sample Python Script (demo.py)

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

Choose your preferred method:

**Terminal:**

```bash
export DEFAULT_LLM_API_KEY="your_api_key"
export DEFAULT_LLM_BASE_URL="your_base_url"
export DEFAULT_LLM_MODEL_NAME="your_model_name"
```

**`.env` file:**

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

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="Example Output">
</div>

## 6. Contributing

We welcome contributions!  Ways to contribute include:

1.  Reporting Issues (Bugs & Errors)
2.  Suggesting Enhancements
3.  Improving Documentation
    *   Fork the repository.
    *   Add your changes.
    *   Submit a pull request.
4.  Writing Code
    *   Fork the repository.
    *   Create a new branch.
    *   Implement your feature or fix.
    *   Submit a pull request.

Check out our documentation for more details: * **[Document](http://oxygent.jd.com/docs/)**

## 7. Community & Support

For any issues, please submit details with reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area. You can also contact the OxyGent Core team via your internal Slack.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="Contact">
</div>

## 8. About the Contributors

Special thanks to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## 9. License

[Apache License](./LICENSE.md)

**OxyGent is provided by Oxygen JD.com.**
**Thank you for your contributions!**