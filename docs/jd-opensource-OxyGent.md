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

<div align="center">
  <a href="https://github.com/jd-opensource/OxyGent/pulls">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome">
  </a>
  <a href="https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="license"/>
  </a>
  <a href="https://pypi.org/project/oxygent/">
    <img src="https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white" alt="pip"/>
  </a>

  <a href="https://github.com/jd-opensource/OxyGent">
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" alt="OxyGent Banner" width="100%"/>
  </a>
</div>

## OxyGent: Build Intelligent Systems Faster with a Modular Python Framework

OxyGent is an open-source Python framework designed to accelerate the development of production-ready intelligent systems, offering a robust and extensible solution for building multi-agent systems.  ([View on GitHub](https://github.com/jd-opensource/OxyGent))

**Key Features:**

*   🚀 **Efficient Development:** Quickly build, deploy, and evolve AI teams with a modular multi-agent framework.  Leverage standardized components for rapid agent assembly.
*   🤝 **Intelligent Collaboration:**  Empower agents with dynamic planning for task decomposition, negotiation, and real-time adaptation. Ensure full auditability of every decision.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies with automated dependency mapping and visual debugging tools for optimized performance.
*   🔁 **Continuous Evolution:**  Built-in evaluation engines automatically generate training data, fostering continuous improvement and transparency.
*   📈 **Scalability:**  A distributed scheduler enables linear cost growth and exponential gains in collaborative intelligence.

## Project Overview

**OxyGent** streamlines the creation of complex, intelligent systems by unifying tools, models, and agents within a modular framework. Developers benefit from transparent, end-to-end pipelines, making building, running, and evolving multi-agent systems a seamless and highly extensible process.

## Framework Core Classes

[Insert the image here:  ![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png)]

## Feature Highlights

*   **For Developers:** Focus on core business logic without reinventing the wheel.
*   **For Enterprises:** Consolidate siloed AI systems into a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork through an intelligent agent ecosystem.

## Quick Start Guide

Get up and running with OxyGent in a few simple steps:

### Step 1: Create and Activate a Python Environment

Choose your preferred method:

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

### Step 2: Install the OxyGent Python Package

Choose your preferred method:

*   **Method 1: Conda**
    ```bash
    pip install oxygent
    ```
*   **Method 2: uv**
    ```bash
    uv pip install oxygent
    ```
*   **Method 3: Development Mode**
    ```bash
    git clone https://github.com/jd-opensource/OxyGent.git
    cd OxyGent
    pip install -r requirements.txt # or use uv
    brew install coreutils # may be required
    ```

### Step 3: (If Using MCP) Install Node.js

Download and install **[Node.js](https://nodejs.org)**.

### Step 4: Example Python Script (demo.py)

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

*   **Method 1: Terminal**
    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

*   **Method 2: .env File**
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

[Insert the image here:  ![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)]

## Contributing

Contribute to OxyGent by:

1.  Reporting Issues (Bugs & Errors)
2.  Suggesting Enhancements
3.  Improving Documentation
    *   Fork the repository
    *   Add your content
    *   Submit a Pull Request
4.  Writing Code
    *   Fork the repository
    *   Create a new branch
    *   Add your feature or improvement
    *   Submit a Pull Request

We welcome all contributions!

For development-related questions, see our documentation: **[Document](http://oxygent.jd.com/docs/)**

## Community & Support

For any issues, please submit reproducible steps and log snippets in the project's Issues area.  You can also contact the OxyGent Core team directly via internal Slack.

[Insert the contact image here: <div align="center"><img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%"></div>]

## About the Contributors

Thanks to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!