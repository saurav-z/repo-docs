# OxyGent: Build Intelligent Systems Faster with This Powerful Python Framework

[OxyGent](https://github.com/jd-opensource/OxyGent) is a cutting-edge Python framework designed to accelerate the development of production-ready intelligent systems, offering a modular, scalable, and collaborative environment for building AI-powered solutions.

**Key Features:**

*   🚀 **Efficient Development:** Build, deploy, and evolve AI teams rapidly with a modular multi-agent framework.
*   🤝 **Intelligent Collaboration:** Leverage dynamic planning for agents to decompose tasks, negotiate solutions, and adapt in real-time.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies and provides automated dependency mapping for optimized performance.
*   🔁 **Continuous Evolution:** Benefit from built-in evaluation engines that generate training data for continuous improvement.
*   📈 **Scalability:** Utilize a distributed scheduler for linear cost growth and exponential gains in collaborative intelligence.

## Why Choose OxyGent?

OxyGent simplifies the entire AI lifecycle, allowing developers to focus on business logic and enabling enterprises to replace siloed systems with a unified framework. Users benefit from seamless teamwork within an intelligent agent ecosystem.

## Project Overview

**OxyGent** is an open-source framework that unifies tools, models, and agents into modular Oxy components. This allows developers to build, run, and evolve multi-agent systems easily and in an infinitely extensible manner.

## Software Architecture

### Architecture Diagram
<!-- Insert architecture diagram here -->
### Architecture Description

*   📦 **Repository:** Organizes agents, tools, LLMs, data, and system files in a unified structure.
*   🛠 **Production Framework:** Provides a complete production chain including registration, building, running, evaluation, and evolution.
*   🖥 **Service Framework:** A full business system server that provides full storage and monitoring support.
*   ⚙️ **Engineering Base:** Includes rich external support with integrated modules like databases and inference engines.

## Feature Highlights

*   **For Developers:** Concentrate on business logic without getting bogged down in infrastructure.
*   **For Enterprises:** Consolidate siloed AI systems into a cohesive, unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork within an intelligent agent ecosystem.

**OxyGent's lifecycle:**

1️⃣ **Code** agents in Python.

2️⃣ **Deploy** with a single command (locally or in the cloud).

3️⃣ **Monitor** every decision made, ensuring full transparency.

4️⃣ **Evolve** your systems automatically using self-improving techniques.

## Quick Start

1.  **Set up your environment:**

    ```bash
    # Using conda
    conda create -n oxy_env python==3.10
    conda activate oxy_env
    pip install oxygent
    ```
    
    ```bash
    # Using uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.10 
    uv venv .venv --python 3.10
    source .venv/bin/activate
    uv pip install oxygent
    ```

2.  **Write and run a sample Python script:**

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

3.  **Configure your LLM settings using environment variables or a `.env` file:**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

    ```bash
    # create a .env file
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

4.  **Run the example script:**

    ```bash
    python demo.py
    ```

5.  **View the output.**

    ![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)

## Contributing

Contribute to OxyGent by:

1.  Reporting issues (bugs & errors).
2.  Suggesting enhancements.
3.  Improving documentation.
4.  Writing code.

See the [documentation](http://oxygent.jd.com/docs/) for detailed instructions.

## Community & Support

For any issues, submit reproducible steps and log snippets in the project's Issues area or contact the OxyGent Core team.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thanks to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!