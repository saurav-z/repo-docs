# OxyGent: Build Production-Ready Intelligent Systems with Ease

**OxyGent empowers developers to build and deploy advanced, production-ready intelligent systems quickly and efficiently.** [Explore the OxyGent Repository](https://github.com/jd-opensource/OxyGent)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

[English](./README.md) | [中文](./README_zh.md)

[<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%" alt="OxyGent Banner">](http://oxygent.jd.com)

## Key Features

*   🚀 **Efficient Development:** Build, deploy, and evolve AI teams rapidly with a modular, multi-agent framework. Standardized components enable rapid assembly and hot-swapping.
*   🤝 **Intelligent Collaboration:** Agents collaborate with dynamic planning, decompose tasks, and adapt to real-time changes, ensuring auditability.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies and simplifies performance optimization across distributed systems.
*   🔁 **Continuous Evolution:** Built-in evaluation engines generate training data, enabling agents to continuously improve.
*   📈 **Scalability:** Distributed scheduler allows for linear cost growth, delivering exponential gains in collaborative intelligence.

## What is OxyGent?

OxyGent is an open-source framework designed to streamline the creation of sophisticated AI systems. It unifies tools, models, and agents into modular Oxy components, providing end-to-end pipelines for building, running, and evolving multi-agent systems. With OxyGent, developers can focus on business logic, enterprises can replace siloed AI systems, and users can experience seamless teamwork from an intelligent agent ecosystem.

## Performance

OxyGent demonstrates competitive performance in the [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) benchmark.

[<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png" width="50%" alt="Performance Graph">](https://huggingface.co/spaces/gaia-benchmark/leaderboard)

## Core Classes

[<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" width="75%" alt="Core Classes Structure">](http://oxygent.jd.com/docs/)

## Quick Start

Follow these steps to get started with OxyGent:

### Step 1: Set up your Python environment
- **Option 1: Conda**

  ```bash
  conda create -n oxy_env python==3.10
  conda activate oxy_env
  ```

- **Option 2: uv**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv python install 3.10
  uv venv .venv --python 3.10
  source .venv/bin/activate
  ```

### Step 2: Install the OxyGent package
- **Option 1: Conda**

  ```bash
  pip install oxygent
  ```

- **Option 2: uv**

  ```bash
  uv pip install oxygent
  ```

- **Option 3: Development Environment**

  ```bash
  git clone https://github.com/jd-opensource/OxyGent.git
  cd OxyGent
  pip install -r requirements.txt # or uv
  brew install coreutils # (May be necessary)
  ```

### Step 3: Node.js Environment (if using MCP)
- Download and install **[Node.js](https://nodejs.org)**

### Step 4: Example Script (demo.py)

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
- **Option 1: Terminal**
  ```bash
  export DEFAULT_LLM_API_KEY="your_api_key"
  export DEFAULT_LLM_BASE_URL="your_base_url"
  export DEFAULT_LLM_MODEL_NAME="your_model_name"
  ```

- **Option 2: .env file**
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

[<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" width="50%" alt="Example Output">](http://oxygent.jd.com/docs/)

## Contributing

We welcome contributions!  Here's how you can help:

1.  Report Issues (Bugs & Errors)
2.  Suggest Enhancements
3.  Improve Documentation:
    *   Fork the repository.
    *   Add your view in document.
    *   Send your pull request.
4.  Write Code:
    *   Fork the repository.
    *   Create a new branch.
    *   Add your feature or improvement.
    *   Send your pull request.

For development-related questions, check the [documentation](http://oxygent.jd.com/docs/).

## Community & Support

Submit reproducible steps and log snippets in the project's Issues area. Contact the OxyGent Core team via your internal Slack.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thank you to all the [contributors](https://github.com/jd-opensource/OxyGent/graphs/contributors)!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!