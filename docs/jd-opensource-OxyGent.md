# OxyGent: Build Production-Ready Intelligent Systems with Ease

**OxyGent empowers developers to build and deploy advanced, production-ready multi-agent systems quickly and efficiently.** [Explore the OxyGent Repository](https://github.com/jd-opensource/OxyGent)

---

## Key Features

*   🚀 **Rapid Development:** Build, deploy, and evolve AI teams with unprecedented efficiency using modular components.
*   🤝 **Intelligent Collaboration:**  Enable dynamic planning, task decomposition, and real-time adaptation for intelligent collaboration.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies, from simple to complex hybrid planning patterns.
*   🔁 **Continuous Evolution:** Built-in evaluation engines generate training data, enabling continuous improvement and transparency.
*   📈 **Scalability:** Distributed scheduler allows for linear cost growth while delivering exponential gains in collaborative intelligence.

## 1. Project Overview

OxyGent is an open-source framework designed to unify tools, models, and agents into modular components. It provides developers with end-to-end pipelines for building, running, and evolving multi-agent systems seamlessly and with limitless extensibility.

## 2. Core Features - Detailed

*   **Efficient Development:** OxyGent's modular multi-agent framework allows for the efficient building, deployment, and evolution of AI teams. Its standardized components enable rapid agent assembly, hot-swapping, and cross-scenario reuse via clean Python interfaces, eliminating the need for complex configurations.
*   **Intelligent Collaboration:** This framework enhances collaboration through dynamic planning, task decomposition, and intelligent solution negotiation. Agents can adapt in real-time to changing conditions, maintaining full auditability of all decisions.
*   **Elastic Architecture:** The underlying architecture supports any agent topology, from simple ReAct agents to complex hybrid planning models. Automated dependency mapping and visual debugging tools facilitate performance optimization across distributed systems.
*   **Continuous Evolution:**  Built-in evaluation engines generate training data automatically. Your agents continuously improve through knowledge feedback loops while maintaining full transparency.
*   **Scalability:** OxyGent's distributed scheduler delivers exponential gains in collaborative intelligence with linear cost growth, effortlessly handling domain-wide optimization and real-time decision-making at scale.

## 3. Software Architecture

### 3.1 Diagram

<!-- Insert architecture diagram here -->

### 3.2 Architecture Description

*   📦 **Repository**: Stores agents, tools, LLMs, data, and system files in a unified structure.
*   🛠 **Production Framework**: Complete production chain including registration, building, running, evaluation, and evolution.
*   🖥 **Service Framework**: Comprehensive business system server, providing full storage and monitoring support.
*   ⚙️ **Engineering Base**: Rich external support, including integrated modules such as databases and inference engines.

## 4. Feature Highlight

**Target Audience:**

*   **For Developers:** Focus on business logic without the overhead of reinventing foundational elements.
*   **For Enterprises:** Replace siloed AI systems with a unified framework to reduce communication overhead.
*   **For Users:** Experience seamless teamwork within an intelligent agent ecosystem.

**Key Lifecycle Stages:**

1.  **Code** agents in Python.
2.  **Deploy** with a single command.
3.  **Monitor** every decision with full transparency.
4.  **Evolve** automatically, enabling self-improving systems.

## 5. Quick Start

**Prerequisites:** Python 3.10 or higher, pip, and an environment to install OxyGent.

**Installation (using conda or uv):**

```bash
# Using conda
conda create -n oxy_env python==3.10
conda activate oxy_env
pip install oxygent

# or (uv)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10 
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install oxygent
```

**Development Environment Setup (optional):**

1.  Download **[Node.js](https://nodejs.org)**
2.  Install required dependencies:

```bash
pip install -r requirements.txt # or in uv
brew install coreutils # maybe essential
```

**Example Python Script:**

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

**LLM Configuration:**

```bash
export DEFAULT_LLM_API_KEY="your_api_key"
export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
export DEFAULT_LLM_MODEL_NAME="your_model_name"  
```

or, create a `.env` file:

```
DEFAULT_LLM_API_KEY="your_api_key"
DEFAULT_LLM_BASE_URL="your_base_url"
DEFAULT_LLM_MODEL_NAME="your_model_name"
```

**Run the Example:**

```bash
python demo.py
```

**Output:**

![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)

## 6. Contributing

We welcome contributions in various forms:

1.  Reporting Issues (Bugs & Errors)
2.  Suggesting Enhancements
3.  Improving Documentation
4.  Writing Code

**Contribution Guidelines:**

1.  Fork the repository.
2.  Create a new branch.
3.  Add your feature or improvement.
4.  Send your pull request.

For development-related issues, refer to the documentation: **[Document](http://oxygent.jd.com/docs/)**

## 7. Community & Support

If you have questions or encounter any issues, please submit reproducible steps and log snippets in the project's Issues area or contact the OxyGent Core team.

**Contact:**

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## 8. About the Contributors

A big thank you to all [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## 9. License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!