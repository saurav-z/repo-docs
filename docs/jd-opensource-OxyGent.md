# OxyGent: Build Production-Ready Intelligent Systems with Ease

**OxyGent is a cutting-edge Python framework designed to accelerate the development of multi-agent systems, enabling developers to create intelligent applications with efficiency and scalability.** ([View on GitHub](https://github.com/jd-opensource/OxyGent))

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

## Key Features of OxyGent

*   🚀 **Efficient Development:** Build, deploy, and evolve AI teams rapidly with a modular multi-agent framework and reusable components.
*   🤝 **Intelligent Collaboration:** Facilitates dynamic planning, task decomposition, and negotiation among agents for robust solutions.
*   🕸️ **Elastic Architecture:** Supports various agent topologies and provides automated dependency mapping and debugging tools.
*   🔁 **Continuous Evolution:** Leverages built-in evaluation engines for auto-generated training data and continuous improvement.
*   📈 **Scalability:** Enables linear cost growth with exponential gains in collaborative intelligence through a distributed scheduler.

## Project Overview

OxyGent is an open-source framework that integrates tools, models, and agents into modular components, simplifying the creation of complex AI systems. It provides end-to-end pipelines for building, running, and evolving multi-agent systems, making the development process seamless and extensible.

## Software Architecture

### Architecture Diagram

<!-- Insert architecture diagram here -->
<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="Architecture Diagram">

### Architecture Description

*   📦 **Repository:** Organizes agents, tools, LLMs, data, and system files.
*   🛠 **Production Framework:** Offers a complete production chain including registration, building, running, evaluation, and evolution.
*   🖥 **Service Framework:** Provides a comprehensive business system server with storage and monitoring.
*   ⚙️ **Engineering Base:** Includes integrated modules such as databases and inference engines, offering robust external support.

## Feature Highlights

*   **For Developers:** Focus on core business logic without redundant development.
*   **For Enterprises:** Unifies siloed AI systems for reduced communication overhead.
*   **For Users:** Provides a seamless teamwork experience from an intelligent agent ecosystem.

**Complete Lifecycle:**

1️⃣ **Code** agents in Python.

2️⃣ **Deploy** with a single command.

3️⃣ **Monitor** every decision transparently.

4️⃣ **Evolve** systems automatically.

## Quick Start

Follow these steps to get started with OxyGent:

1.  **Set up your environment:**
    *   Create and activate a Python environment (using conda):
        ```bash
        conda create -n oxy_env python==3.10
        conda activate oxy_env
        ```
        or (using uv):
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv python install 3.10 
        uv venv .venv --python 3.10
        source .venv/bin/activate
        ```
2.  **Install OxyGent:**
    *   Using conda:
        ```bash
        pip install oxygent
        ```
        or (using uv):
        ```bash
        uv pip install oxygent
        ```
3.  **Development Environment Setup**
    *   Download **[Node.js](https://nodejs.org)**
    *   Download our requirements:
       ```bash
       pip install -r requirements.txt # or in uv
       brew install coreutils # maybe essential
       ```
4.  **Write a sample Python script:**
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
5.  **Configure your LLM settings (e.g., OpenAI API key):**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
    export DEFAULT_LLM_MODEL_NAME="your_model_name"  
    ```

    Alternatively, set them in a `.env` file:

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
    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="Output">

## Contributing

We welcome contributions to OxyGent! Here's how you can contribute:

1.  Report issues (bugs and errors).
2.  Suggest enhancements.
3.  Improve documentation.
    *   Fork the repository.
    *   Add your view in the document.
    *   Send your pull request.
4.  Write code.
    *   Fork the repository.
    *   Create a new branch.
    *   Add your feature or improvement.
    *   Send your pull request.

For any development-related issues, refer to our documentation:  [Document](http://oxygent.jd.com/docs/)

## Community & Support

If you face any problems, please submit reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area or contact the OxyGent Core team.

Welcome to contact us:

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thanks to all contributors!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!