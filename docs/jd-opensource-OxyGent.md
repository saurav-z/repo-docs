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

# OxyGent: Build Production-Ready Intelligent Systems with Ease

[OxyGent](https://github.com/jd-opensource/OxyGent) is an open-source Python framework designed to streamline the development of intelligent systems, enabling developers to rapidly build and deploy sophisticated, production-ready AI applications.

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%"/>
</p>

<p align="center">
  Visit our website: <a href="http://oxygent.jd.com">OxyGent</a>
</p>

## Key Features

*   **Efficient Development:** Build, deploy, and evolve AI teams with unprecedented efficiency using modular components for rapid assembly.
*   **Intelligent Collaboration:** Leverage dynamic planning paradigms for intelligent task decomposition, negotiation, and real-time adaptation.
*   **Elastic Architecture:** Supports any agent topology, with automated dependency mapping and visual debugging tools.
*   **Continuous Evolution:** Built-in evaluation engines generate training data for continuous agent improvement, with full transparency.
*   **Scalability:**  Distributed scheduler enables linear cost growth while delivering exponential gains in collaborative intelligence, optimizing domain-wide decision making.

## Project Overview

OxyGent is a cutting-edge framework that simplifies the creation of complex, multi-agent systems. By unifying tools, models, and agents within modular Oxy components, developers gain access to transparent, end-to-end pipelines that facilitate seamless building, running, and evolution of intelligent systems.

## Framework Core Classes

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" width="80%"/>
</p>

## Feature Highlights

*   **For Developers:** Focus on business logic without reinventing the wheel.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

## Quick Start

Get started with OxyGent in a few easy steps:

### Step 1: Create and activate a Python environment

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

### Step 2: Install the required Python package

*   **Method 1: conda**
    ```bash
    pip install oxygent
    ```
*   **Method 2: uv**
    ```bash
    uv pip install oxygent
    ```
*   **Method 3: Set up a development environment**
    ```bash
    git clone https://github.com/jd-opensource/OxyGent.git
    cd OxyGent
    pip install -r requirements.txt  # or in uv
    brew install coreutils # maybe essential
    ```

### Step 3: Node.js Environment (if using MCP)

*   Download and install **[Node.js](https://nodejs.org)**

### Step 4: Write a sample Python script (`demo.py`)

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

*   **Method 1: Declare in terminal**
    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```
*   **Method 2: Create a `.env` file**
    ```bash
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

### Step 6: Run the example

```bash
python demo.py
```

### Step 7: View the output

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" width="80%"/>
</p>

## Contributing

We welcome contributions! Please see our contribution guidelines:

1.  **Report Issues:** Report bugs and errors.
2.  **Suggest Enhancements:** Suggest improvements to the framework.
3.  **Improve Documentation:** Fork the repository, add your view in document and send your pull request.
4.  **Write Code:** Fork the repository, create a new branch, add your feature or improvement, and send your pull request.

For development-related problems, please check the [documentation](http://oxygent.jd.com/docs/).

## Community & Support

If you encounter any issues, submit reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area. Or contact the OxyGent Core team directly via your internal Slack.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

A big thanks to all the contributors!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!
```
Key improvements and explanations:

*   **SEO Optimization:**  Includes keywords like "Python framework," "intelligent systems," "AI," "multi-agent," and "open-source" in headings and text to improve searchability.  The one-sentence hook clearly communicates the value proposition.
*   **Clear Structure:** Uses clear headings (H2, H3) for better readability and organization.  Each section is logically separated.
*   **Concise Language:**  Rewords sentences for improved clarity and brevity. Avoids jargon where possible.
*   **Emphasis on Benefits:** Highlights the benefits of using OxyGent (e.g., efficiency, collaboration, scalability).
*   **Actionable Quick Start:** Provides a step-by-step guide with clear instructions, code samples, and environment setup.
*   **Visuals:**  Includes image links to enhance the visual appeal of the README.
*   **Call to Action:**  Encourages contributions.
*   **Comprehensive Information:** Retains all the original content while improving its presentation and clarity.
*   **Links to repo and resources:** Includes links to the original repo and documentation.
*   **Improved Formatting:** Uses bolding, lists, and code blocks effectively for readability.
*   **Contributor Section:**  Maintains and improves the contributor section.