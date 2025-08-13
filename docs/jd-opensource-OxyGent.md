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
</p>

<h1 align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%" alt="OxyGent Banner">
</h1>
<h2 align="center">OxyGent: Build intelligent systems faster with a modular, collaborative AI framework.</h2>
<p align="center">
  <a href="http://oxygent.jd.com">OxyGent Website</a>
</p>

---

## What is OxyGent?

**OxyGent** is an open-source, advanced Python framework designed to accelerate the development of production-ready intelligent systems. It unifies tools, models, and agents into a modular structure, providing developers with end-to-end pipelines for building, running, and evolving multi-agent systems.

## Key Features

*   🚀 **Efficient Development:** Build, deploy, and evolve AI teams with unparalleled speed. Standardized components and clean Python interfaces enable rapid agent assembly, hot-swapping, and cross-scenario reuse.
*   🤝 **Intelligent Collaboration:** Empower agents with dynamic planning, task decomposition, negotiation, and real-time adaptation.  OxyGent agents handle emergent challenges naturally while maintaining full auditability.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies, from simple ReAct to complex hybrid patterns. Benefit from automated dependency mapping and visual debugging tools for performance optimization.
*   🔁 **Continuous Evolution:** Built-in evaluation engines generate training data, allowing agents to continuously improve through knowledge feedback loops while maintaining full transparency.
*   📈 **Scalability:** Scale your collaborative intelligence linearly with OxyGent's distributed scheduler, which handles domain-wide optimization and real-time decision-making at any scale.

## Performance

The latest version of OxyGent (July 15, 2025) achieved 59.14 points in the [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) benchmark.

<p align="center">
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png" alt="Performance Graph">
</p>

## Architecture Overview

### 3.1 Diagram
<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="OxyGent Architecture Diagram">
</p>

### 3.2 Architecture Components

*   📦 **Repository:** Stores agents, tools, LLMs, data, and system files.
*   🛠 **Production Framework:** Manages the complete production chain, including registration, building, running, evaluation, and evolution.
*   🖥 **Service Framework:** Provides a complete business system server with storage and monitoring support.
*   ⚙️ **Engineering Base:** Includes integrated modules such as databases and inference engines.

## Benefits

*   **For Developers:** Focus on business logic without reinventing the wheel.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

## Quick Start

Get up and running with OxyGent in minutes:

1.  **Set up your environment:**
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

3.  **Alternatively, for development:**
    *   Download **[Node.js](https://nodejs.org)**
    *   Install requirements:
        ```bash
        pip install -r requirements.txt
        brew install coreutils # maybe essential
        ```

4.  **Write and run a sample script (demo.py):**

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
    or create a `.env` file:
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

    <p align="center">
      <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="Output">
    </p>

## Contributing

We welcome contributions!  You can contribute by:

*   Reporting issues (bugs and errors)
*   Suggesting enhancements
*   Improving documentation
    *   Fork the repository
    *   Add your view in document
    *   Send your pull request
*   Writing code
    *   Fork the repository
    *   Create a new branch
    *   Add your feature or improvement
    *   Send your pull request

Check out the documentation: * **[Document](http://oxygent.jd.com/docs/)**

## Community & Support

For any issues, submit reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area or contact the OxyGent Core team.

<p align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="Contact Information" width="50%">
</p>

## About the Contributors

Thank you to all the [contributors](https://github.com/jd-opensource/OxyGent/graphs/contributors)!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" alt="Contributors">
</a>

## License

[Apache License](./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!
```
Key improvements and changes:

*   **SEO Optimization:** Added headings, subheadings, and keywords like "Python framework," "multi-agent," "AI," and "intelligent systems" to improve search engine visibility.
*   **Concise Hook:**  Created a compelling one-sentence hook to grab attention.
*   **Clear Structure:**  Used bullet points and numbered lists for easy readability.
*   **Benefit-Oriented:**  Emphasized the benefits for developers, enterprises, and users.
*   **Improved Quick Start:**  Made the Quick Start section clearer, providing both `conda` and `uv` installation options, and separated the setup, installation, and running steps.
*   **Contributor Visualization:**  Kept the contributors section and added an image to show contributors.
*   **Call to Action:** Encouraged contributions with clear instructions.
*   **Removed redundant information**: Streamlined text and removed unnecessary introductions.
*   **Added Alt text to Images**: Improved accessibility and SEO.