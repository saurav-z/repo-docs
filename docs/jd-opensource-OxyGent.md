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

<h1 align="center">OxyGent: Build Intelligent Systems Faster with a Powerful Python Framework</h1>

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%" alt="OxyGent Banner">
  <p>Empowering developers, OxyGent is an advanced Python framework for building production-ready multi-agent systems.</p>
  <a href="http://oxygent.jd.com">
    <img src="https://img.shields.io/website?label=Website&url=http%3A%2F%2Foxygent.jd.com&color=blue" alt="OxyGent Website">
  </a>
</div>

## Key Features

*   **Efficient Development:** Build, deploy, and evolve AI teams with unprecedented speed using modular components and clean Python interfaces.
*   **Intelligent Collaboration:** Leverage dynamic planning paradigms for intelligent task decomposition, negotiation, and adaptation.
*   **Elastic Architecture:** Supports any agent topology, providing automated dependency mapping and visual debugging for optimal performance.
*   **Continuous Evolution:** Agents improve through knowledge feedback loops with built-in evaluation engines that auto-generate training data.
*   **Scalability:** Achieve exponential gains in collaborative intelligence with a distributed scheduler that handles domain-wide optimization at any scale.

## 1. Project Overview

**OxyGent** is an open-source Python framework designed to streamline the development of intelligent systems. It unifies tools, models, and agents into modular Oxy components, providing developers with end-to-end pipelines.  This framework allows for the seamless building, running, and evolving of multi-agent systems, making the process infinitely extensible.

## 2. Core Features (Expanded)

*   **Modular Design:**  OxyGent components are designed like LEGO bricks, allowing for rapid agent assembly. Support for hot-swapping and cross-scenario reuse saves time and resources.
*   **Dynamic Planning:** Agents intelligently collaborate by decomposing tasks, negotiating solutions, and adapting to changes in real-time.  This reduces the reliance on rigid workflows and provides full auditability.
*   **Flexible Architecture:**  Supports a wide range of agent topologies, from simple to complex hybrid planning patterns, and includes automated dependency mapping and visual debugging tools.
*   **Iterative Improvement:**  Agents continuously learn and improve through feedback loops, utilizing built-in evaluation engines to auto-generate training data.
*   **Scalable Design:** The distributed scheduler facilitates linear cost growth while providing exponential gains in collaborative intelligence.

## 3. Performance

The latest version of OxyGent (July 15, 2025) in the [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) get 59.14 points, and current top opensource system OWL gets 60.8 points.

![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png)

## 4. Framework Core Classes

![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png)

## 5. Use Cases

OxyGent excels in the following areas:

*   **For Developers:** Focus on core business logic without the need to reinvent the wheel.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, minimizing communication overhead.
*   **For Users:** Experience intelligent teamwork from a sophisticated agent ecosystem.

## 6. Quick Start

Get started with OxyGent in a few simple steps:

### Step 1: Set up your environment

Choose your preferred environment setup:

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

Choose your preferred installation method:

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

### Step 3: Node.js Requirement (if using MCP)

*   Download and install **[Node.js](https://nodejs.org)**

### Step 4: Create a sample script (demo.py)

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

Choose your preferred way to set environment variables:

*   **Method 1: Terminal**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"
    export DEFAULT_LLM_MODEL_NAME="your_model_name"  
    ```
*   **Method 2: .env file**

    ```bash
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

### Step 6: Run the example

```bash
python demo.py
```

### Step 7: View Output

The output will look like this:

![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)

## 7. Contributing

We welcome contributions!  Here's how you can get involved:

1.  **Report Issues:**  Help us by reporting bugs and errors.
2.  **Suggest Enhancements:**  Share your ideas for improving OxyGent.
3.  **Improve Documentation:** Contribute to our documentation by forking the repository, adding your view in document and sending a pull request.
4.  **Write Code:**  Contribute new features or improvements by forking the repository, creating a new branch, adding your feature or improvement, and sending a pull request.

For development-related questions, please consult the [OxyGent Documentation](http://oxygent.jd.com/docs/).

## 8. Community & Support

If you encounter any issues, please submit reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area, or contact the OxyGent Core team through your internal Slack.

We encourage you to connect with us.
<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## 9. About the Contributors

Thanks to all the following contributors who have helped shape OxyGent.

[<img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />](https://github.com/jd-opensource/OxyGent/graphs/contributors)

## 10. License

OxyGent is licensed under the [Apache License](./LICENSE.md).

#### OxyGent is provided by Oxygen JD.com
#### Thank you for your contributions!

**[Back to Top](#oxygent-build-intelligent-systems-faster-with-a-powerful-python-framework)**
```
Key improvements and SEO optimizations:

*   **Compelling Headline:** A strong and keyword-rich headline to attract users.
*   **Concise Summary:** A one-sentence hook to grab attention.
*   **Detailed Feature List:** Expanded descriptions for each feature.
*   **Clear Headings:**  Organized the content using clear and descriptive headings.
*   **Actionable Quick Start:** Added more details about each step in Quick Start.
*   **Community and Support Emphasis:** Clearly states how to seek help.
*   **Back to Top Link:** Added an anchor to make it easy for users to jump to the top of the page.
*   **Keyword Optimization:** Used keywords like "Python framework," "intelligent systems," and "multi-agent systems" throughout the document to improve search engine visibility.
*   **Enhanced Formatting:** Used bolding, lists, and other formatting to improve readability.
*   **Clear Call to Action:** Encourages contributions and highlights documentation and community support.
*   **Comprehensive Information:** Provided a more complete and informative overview of the project.
*   **Removed unnecessary HTML.**
*   **Improved Structure:** Simplified the structure for easier navigation.
*   **Website and Support Links:** Included links to the website and issue tracker to drive traffic and provide support options.
*   **Contributor Information:** Added a "Contributors" section, making it easier for users to learn more about the project developers.