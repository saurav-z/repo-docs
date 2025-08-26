# OxyGent: Build Intelligent Systems with Ease

**OxyGent is a powerful, open-source Python framework designed to streamline the development of production-ready, multi-agent intelligent systems.**  [Explore the OxyGent Repo](https://github.com/jd-opensource/OxyGent)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%">
</p>

<p align="center">
  Visit our website: <a href="http://oxygent.jd.com">OxyGent</a>
</p>

## Key Features of OxyGent

*   🚀 **Efficient Development:** Build, deploy, and evolve AI teams with unmatched speed. Standardized, modular components enable rapid agent assembly through clean Python interfaces.
*   🤝 **Intelligent Collaboration:**  Supercharges collaboration with dynamic planning, task decomposition, and real-time adaptation capabilities for emergent challenges.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies, from ReAct to complex hybrid patterns, with automated dependency mapping for optimized performance across distributed systems.
*   🔁 **Continuous Evolution:** Built-in evaluation engines generate training data, fostering continuous agent improvement and knowledge feedback loops.
*   📈 **Scalability:**  A distributed scheduler enables linear cost growth with exponential gains in collaborative intelligence, optimizing performance at any scale.

## What's New

OxyGent demonstrates strong performance, achieving 59.14 points on the [GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard), showcasing its competitive capabilities.

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png" alt="GAIA Benchmark Score" width="50%">
</p>

## Core Framework Components

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="Framework Structure" width="80%">
</p>

## Who Can Benefit?

*   **Developers:** Focus on business logic without reinventing the wheel.
*   **Enterprises:** Replace siloed AI systems with a unified and efficient framework, reducing communication overhead.
*   **Users:** Experience seamless teamwork from an intelligent agent ecosystem.

## Quick Start Guide

Follow these steps to get started with OxyGent:

### Step 1: Create and Activate a Python Environment

*   **Using Conda:**

    ```bash
    conda create -n oxy_env python==3.10
    conda activate oxy_env
    ```

*   **Using uv:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.10 
    uv venv .venv --python 3.10
    source .venv/bin/activate
    ```

### Step 2: Install the OxyGent Package

*   **Using Conda:**

    ```bash
    pip install oxygent
    ```

*   **Using uv:**

    ```bash
    uv pip install oxygent
    ```

*   **For Development:**

    ```bash
    git clone https://github.com/jd-opensource/OxyGent.git
    cd OxyGent
    pip install -r requirements.txt # or use uv
    brew install coreutils # if needed
    ```

### Step 3: Install Node.js (if using MCP)

*   Download and install **[Node.js](https://nodejs.org)**

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

### Step 6: Run the Example

```bash
python demo.py
```

### Step 7: View the Output

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="Example Output" width="70%">
</p>

## Contributing

We welcome contributions! Here's how you can get involved:

1.  **Report Issues:**  Report bugs and errors.
2.  **Suggest Enhancements:** Propose new features.
3.  **Improve Documentation:** Fork the repository, add your contributions, and submit a pull request.
4.  **Write Code:** Fork the repository, create a branch, add your feature or improvement, and submit a pull request.

For development help, please check our documentation: * **[Document](http://oxygent.jd.com/docs/)**

## Community & Support

Encountering issues?  Submit reproducible steps and log snippets in the project's Issues area. Contact the OxyGent Core team via internal Slack.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thank you to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License](./LICENSE.md)

**OxyGent is proudly provided by Oxygen JD.com.**
**Thank you for your contributions!**