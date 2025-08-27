# OxyGent: Build Intelligent Systems Faster with a Modular Python Framework

**OxyGent is a cutting-edge, open-source Python framework enabling developers to swiftly build, deploy, and evolve sophisticated, production-ready intelligent systems.** [Explore the original repository](https://github.com/jd-opensource/OxyGent)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

[<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%">](http://oxygent.jd.com)

## Key Features

*   🚀 **Efficient Development:** Build, deploy, and evolve AI teams rapidly with a modular multi-agent framework, standardized components, and clean Python interfaces.
*   🤝 **Intelligent Collaboration:** Facilitate dynamic planning, task decomposition, negotiation, and real-time adaptation among agents.
*   🕸️ **Elastic Architecture:** Support various agent topologies and optimize performance across distributed systems with automated dependency mapping and visual debugging tools.
*   🔁 **Continuous Evolution:** Enable agents to continuously improve through knowledge feedback loops with built-in evaluation engines and maintain full transparency.
*   📈 **Scalability:** Achieve linear cost growth with exponential gains in collaborative intelligence through OxyGent's distributed scheduler.

## Why Choose OxyGent?

*   **For Developers:** Focus on your business logic instead of reinventing the wheel.
*   **For Enterprises:** Unify siloed AI systems and reduce communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

## Quick Start Guide

Get up and running with OxyGent in a few simple steps:

### Step 1: Create and Activate a Python Environment

Choose one of the following methods:

*   **Using Conda:**

    ```bash
    conda create -n oxy_env python==3.10
    conda activate oxy_env
    ```

*   **Using `uv`:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.10
    uv venv .venv --python 3.10
    source .venv/bin/activate
    ```

### Step 2: Install the OxyGent Package

Choose one of the following methods:

*   **Using Conda:**

    ```bash
    pip install oxygent
    ```

*   **Using `uv`:**

    ```bash
    uv pip install oxygent
    ```

*   **For Development (from source):**

    ```bash
    git clone https://github.com/jd-opensource/OxyGent.git
    cd OxyGent
    pip install -r requirements.txt # or use uv
    brew install coreutils # maybe essential
    ```

### Step 3: Node.js Environment (if using MCP)

*   Download and install **[Node.js](https://nodejs.org)**.

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

*   **Declare in Terminal:**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

*   **Create a `.env` File:**

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
<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="Output Example" width="80%">

## Contributing

We welcome your contributions! Here's how you can help:

1.  **Report Issues:**  Report bugs and errors.
2.  **Suggest Enhancements:** Propose new features and improvements.
3.  **Improve Documentation:** Fork the repository, add your documentation updates, and submit a pull request.
4.  **Write Code:** Fork the repository, create a new branch for your feature or improvement, and submit a pull request.

Please check our [Document](http://oxygent.jd.com/docs/) for more details.

## Community & Support

For any issues or questions, please submit them in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area or contact the OxyGent Core team.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thanks to all the following [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.
<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License](./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!