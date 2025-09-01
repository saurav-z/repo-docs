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
[![PyPI](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%" alt="OxyGent Banner">
</div>

<h2 align="center">Build intelligent systems quickly and efficiently with **OxyGent**, a cutting-edge Python framework.</h2>

<div align="center">
  <a href="http://oxygent.jd.com">
    OxyGent Website
  </a>
</div>

## Key Features

*   🚀 **Rapid Development:** OxyGent's modular design lets you rapidly build, deploy, and evolve multi-agent AI systems.
*   🤝 **Intelligent Collaboration:** Leverage dynamic planning for agents to decompose tasks, negotiate solutions, and adapt in real-time.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies, enabling flexible and scalable system design.
*   🔁 **Continuous Evolution:** Built-in evaluation engines provide continuous improvement through knowledge feedback loops.
*   📈 **Scalability:** The distributed scheduler allows for linear cost growth while delivering exponential gains in collaborative intelligence.

## Overview of OxyGent

**OxyGent** is an open-source Python framework designed to streamline the development of production-ready intelligent systems. By unifying tools, models, and agents within a modular architecture, OxyGent empowers developers with transparent, end-to-end pipelines. This framework makes building, running, and evolving multi-agent systems seamless and highly extensible.  Visit the [OxyGent GitHub repository](https://github.com/jd-opensource/OxyGent) for the source code and further information.

## Core Classes & Architecture

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" width="80%" alt="OxyGent Architecture">
</div>

## Why Use OxyGent?

*   **For Developers:** Focus on business logic rather than infrastructure.
*   **For Enterprises:** Replace siloed AI systems with a unified, streamlined framework.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

## Quick Start Guide

Get started with OxyGent in a few simple steps:

### Step 1: Set Up Your Python Environment

Choose your preferred method for managing the Python environment:

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

### Step 2: Install OxyGent

*   **Method 1: pip**

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
    pip install -r requirements.txt  # or using uv
    brew install coreutils # essential
    ```

### Step 3: Node.js (if using MCP)

*   Download and install **[Node.js](https://nodejs.org)**.

### Step 4: Sample Python Script (demo.py)

Create a demo Python script to get started:

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

### Step 5: Configure Environment Variables

*   **Method 1: Terminal**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

*   **Method 2: .env File**

    Create a `.env` file in the same directory as your Python script with the following content:

    ```
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

### Step 6: Run the Example

```bash
python demo.py
```

### Step 7: View the output

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" width="80%" alt="OxyGent Demo Output">
</div>

## Contributing to OxyGent

We welcome contributions! Here's how you can help:

1.  **Report Issues:**  Submit bug reports and error descriptions.
2.  **Suggest Enhancements:**  Propose new features and improvements.
3.  **Improve Documentation:**  Fork the repository, add your views in the document, and send a pull request.
4.  **Write Code:**  Fork the repository, create a new branch, add your feature or improvement, and submit a pull request.

For development-related questions, consult our comprehensive [documentation](http://oxygent.jd.com/docs/).

## Community & Support

If you encounter any issues, please submit detailed, reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area.  Or, contact the OxyGent Core team via internal Slack.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## Contributors

Thanks to all contributors!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

This project is licensed under the [Apache License](./LICENSE.md).

#### OxyGent is provided by Oxygen JD.com

#### Thanks for your Contributions!
```
Key improvements and optimizations:

*   **SEO Optimization:** Included relevant keywords like "Python framework," "multi-agent," "AI," "intelligent systems," "rapid development," etc. in headings and descriptions.
*   **Concise Hook:**  A compelling one-sentence hook to grab the reader's attention.
*   **Clear Structure:**  Well-defined sections with clear headings and subheadings for readability.
*   **Bulleted Key Features:**  Easy-to-scan list of core capabilities.
*   **Actionable Quick Start:** Detailed, step-by-step instructions for getting started.
*   **Visual Enhancements:** Included images to make the README visually appealing.
*   **Community and Support:** Added sections to provide support details.
*   **Direct Links:**  Provided direct links to the GitHub repository, documentation, and the website.
*   **Consistent Formatting:**  Used consistent Markdown formatting for better readability.
*   **License Information:**  Clearly stated the project's license.
*   **Contact details:** Added contact information to the community support.