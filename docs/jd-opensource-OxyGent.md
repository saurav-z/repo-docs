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

<html>
    <h2 align="center">
      <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="1256"/>
    </h2>
    <h3 align="center">
      An advanced Python framework that empowers developers to quickly build production-ready intelligent systems. 
    </h3>
    <h3 align="center">
      Visit our website:
      <a href="http://oxygent.jd.com">OxyGent</a>
    </h3>
</html>

## 1. Project Overview
**OxyGent** is an open-source framework that unifies tools, models, and agents into modular Oxy. Empowering developers with transparent, end-to-end pipelines, OxyGent makes building, running, and evolving multi-agent systems seamless and infinitely extensible.

## 2. Core Features
🏎️ **Efficient Development**
- OxyGent is a modular multi-agent framework that lets you build, deploy, and evolve AI teams with unprecedented efficiency. Its standardized Oxy components snap together like LEGO bricks, enabling rapid assembly of agents while supporting hot-swapping and cross-scenario reuse - all through clean Python interfaces without messy configs.

🤝 **Intelligent Collaboration**
- The framework supercharges collaboration with dynamic planning paradigms, where agents intelligently decompose tasks, negotiate solutions, and adapt to changes in real-time. Unlike rigid workflow systems, OxyGent's agents handle emergent challenges naturally while maintaining full auditability of every decision.

🕸️ **Elastic Architecture**
- Under the hood, an elastic architecture supports any agent topology- from simple ReAct to complex hybrid planning patterns. Automated dependency mapping and visual debugging tools make it easy to optimize performance across distributed systems.

🔁 **Continuous Evolution**
- Every interaction becomes a learning opportunity - thanks to built-in evaluation engines that auto-generate training data. Your agents continuously improve through knowledge feedback loops while maintaining full transparency.

📈 **Scalability**
- Scaling follows Metcalfe's Law- OxyGent's distributed scheduler enables linear cost growth while delivering exponential gains in collaborative intelligence. The system effortlessly handles domain-wide optimization and real-time decision making at any scale.

The latest version of OxyGent (July 15, 2025) in the [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) get 59.14 points, and current top opensource system OWL gets 60.8 points.

![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png)

## 3. Framework Core Classes
![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png)

## 4. Feature Highlight
**For Developers**: Focus on business logic without reinventing the wheel.

**For Enterprises**: Replace siloed AI systems with a unified framework, reducing communication overhead.

**For Users**: Experience seamless teamwork from an intelligent agent ecosystem.

## 5. Quick Start
### Step 1: Create and activate a python environment
- Method 1: conda
   ```bash
   conda create -n oxy_env python==3.10
   conda activate oxy_env
   ```
- Method 2: uv
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.10 
   uv venv .venv --python 3.10
   source .venv/bin/activate
   ```
### Step 2: Install the required python package
- Method 1: conda
   ```bash
   pip install oxygent
   ```
- Method 2: uv
   ```bash
   uv pip install oxygent
   ```
- Method 3: set develop environment
   ```bash
   git clone https://github.com/jd-opensource/OxyGent.git
   cd OxyGent
   pip install -r requirements.txt # or in uv
   brew install coreutils # maybe essential
   ```
### Step 3: Node.js Environment (if using MCP)
- Download and install **[Node.js](https://nodejs.org)**

### Step 4: Write a sample python script
- demo.py
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
- Method 1: Declare in terminal
   ```bash
   export DEFAULT_LLM_API_KEY="your_api_key"
   export DEFAULT_LLM_BASE_URL="your_base_url"
   export DEFAULT_LLM_MODEL_NAME="your_model_name"  
   ```
- Method 2: Create a .env file
   ```bash
   DEFAULT_LLM_API_KEY="your_api_key"
   DEFAULT_LLM_BASE_URL="your_base_url"
   DEFAULT_LLM_MODEL_NAME="your_model_name"
   ```
### Step 6: Run the example
- Start the multi-agent system
   ```bash
   python demo.py
   ```
### Step 7: View the output
- ![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)

## 6. Contributing
There are several ways you can contribute to OxyGent:

1. Reporting Issues (Bugs & Errors)
2. Suggesting Enhancements
3. Improving Documentation
    - Fork the repository
    - Add your view in document
    - Send your pull request
4. Writing Code
    - Fork the repository
    - Create a new branch
    - Add your feature or improvement
    - Send your pull request

We appreciate all kinds of contributions! 🎉🎉🎉
If you have problems about development, please check our document: * **[Document](http://oxygent.jd.com/docs/)**

## 7. Community & Support
If you encounter any issues along the way, you are welcomed to submit reproducible steps and log snippets in the project's Issues area, or contact the OxyGent Core team directly via your internal Slack.

Welcome to contact us:

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>


## 8. About the Contributors
Thanks to all the following [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.
<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## 9. License
[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com 
#### Thanks for your Contributions!