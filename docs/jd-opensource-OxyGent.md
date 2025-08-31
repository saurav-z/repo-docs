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
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" alt="OxyGent Banner" width="100%">
  </a>
  <h3><a href="http://oxygent.jd.com">OxyGent</a>: Build intelligent systems faster with this advanced Python framework.</h3>
</div>

## Key Features

*   🚀 **Efficient Development:** Quickly build, deploy, and evolve AI teams with a modular, LEGO-brick-like framework.
*   🤝 **Intelligent Collaboration:** Enables dynamic planning, task decomposition, and real-time adaptation for sophisticated agent interactions.
*   🕸️ **Elastic Architecture:** Supports various agent topologies with automated dependency mapping and visual debugging.
*   🔁 **Continuous Evolution:** Built-in evaluation engines provide knowledge feedback loops and continuous agent improvement.
*   📈 **Scalability:** A distributed scheduler enables linear cost growth while delivering exponential gains in collaborative intelligence.

## 1. Project Overview

**OxyGent** is an open-source, Python-based framework that streamlines the development of production-ready intelligent systems. It unifies tools, models, and agents into modular components, enabling developers to create, deploy, and evolve multi-agent systems with unprecedented efficiency.

## 2. Core Features Explained

*   **Efficient Development:** OxyGent's modular architecture facilitates rapid assembly of agents. Standardized components allow for hot-swapping and cross-scenario reuse through clean Python interfaces.
*   **Intelligent Collaboration:** The framework encourages collaborative problem-solving through dynamic planning, task decomposition, and negotiation among agents. Full auditability ensures transparency.
*   **Elastic Architecture:** Supports diverse agent topologies, from simple ReAct to complex hybrid planning patterns. Automated dependency mapping and visual debugging tools are provided for optimized performance.
*   **Continuous Evolution:** Built-in evaluation engines generate training data automatically. Agents continuously improve through knowledge feedback loops, with full transparency.
*   **Scalability:** OxyGent's distributed scheduler ensures linear cost growth with exponential gains in collaborative intelligence, making it suitable for domain-wide optimization and real-time decision-making at any scale.

## 3. Framework Core Classes

<div align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="OxyGent Architecture">
</div>

## 4. Feature Highlights & Benefits

*   **For Developers:** Focus on core business logic without the need to build everything from scratch.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, streamlining communication.
*   **For Users:** Experience the benefits of seamless teamwork within an intelligent agent ecosystem.

## 5. Quick Start

Follow these steps to get started with OxyGent:

### Step 1: Set Up Your Python Environment

Choose one of the following methods:

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

Choose one of the following methods:

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

### Step 3: Node.js Environment (if using MCP)

*   Download and install **[Node.js](https://nodejs.org)**

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

Choose one of the following methods:

*   **Method 1: Declare in Terminal**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

*   **Method 2: Create a `.env` File**

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
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="OxyGent Output Example">
</div>

## 6. Contributing

We welcome contributions of all kinds!

*   **Report Issues:**  Report bugs and errors.
*   **Suggest Enhancements:** Propose new features or improvements.
*   **Improve Documentation:** Fork the repository, add your views, and submit a pull request.
*   **Write Code:** Fork the repository, create a new branch, add your feature or improvement, and send a pull request.

For development questions, see our [Documentation](http://oxygent.jd.com/docs/).

## 7. Community & Support

If you have any issues, please submit reproducible steps and log snippets in the project's Issues area or contact the OxyGent Core team.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="Contact">
</div>

## 8. About the Contributors

Thank you to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" alt="Contributors">
</a>

## 9. License

[Apache License]( ./LICENSE.md)

---

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!