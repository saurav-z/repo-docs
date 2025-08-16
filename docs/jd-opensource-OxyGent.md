# OxyGent: Build Production-Ready Intelligent Systems with Ease

[OxyGent](https://github.com/jd-opensource/OxyGent) is an open-source Python framework that empowers developers to rapidly build and deploy advanced multi-agent systems for production environments.

---

## Key Features

*   🚀 **Rapid Development:** Build, deploy, and evolve AI teams with unprecedented efficiency using modular, reusable components.
*   🤝 **Intelligent Collaboration:** Facilitate dynamic task decomposition, negotiation, and real-time adaptation among agents.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies, optimized performance across distributed systems, and visual debugging.
*   🔁 **Continuous Evolution:** Leverage built-in evaluation engines for automated training data generation and transparent knowledge feedback loops.
*   📈 **Scalability:** Achieve linear cost growth with exponential gains in collaborative intelligence via a distributed scheduler.

## 1. Project Overview

OxyGent provides a unified framework for building, running, and evolving multi-agent systems. It integrates tools, models, and agents into modular Oxy components.

## 2. Core Features (Detailed)

*   **Efficient Development:** The framework utilizes modular components allowing developers to quickly assemble agents through simple Python interfaces. This design supports hot-swapping and cross-scenario reuse.
*   **Intelligent Collaboration:** Offers dynamic planning capabilities, enabling agents to autonomously decompose tasks, negotiate solutions, and adapt to environmental changes while maintaining a fully auditable decision-making process.
*   **Elastic Architecture:** Provides support for a wide variety of agent topologies and incorporates tools for automated dependency mapping and visual debugging.
*   **Continuous Evolution:** Built-in evaluation engines automatically generate training data, allowing agents to continuously improve through knowledge feedback loops.
*   **Scalability:** Includes a distributed scheduler enabling linear cost growth and exponential gains in collaborative intelligence.

The latest version of OxyGent (July 15, 2025) achieves 59.14 points on the [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) benchmark.

## 3. Software Architecture

### 3.1 Diagram

<!-- Insert architecture diagram here -->

### 3.2 Architecture Description

*   📦 **Repository:** A unified storage solution for agents, tools, LLMs, data, and system files.
*   🛠 **Production Framework:** Provides a comprehensive production chain, including registration, building, running, evaluation, and evolution capabilities.
*   🖥 **Service Framework:** Offers a complete business system server with comprehensive storage and monitoring support.
*   ⚙️ **Engineering Base:** Provides rich external support with integrated modules, such as databases and inference engines.

## 4. Feature Highlight

*   **For Developers:** Focus on core business logic, reducing the need to reinvent the wheel.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, minimizing communication overhead.
*   **For Users:** Experience seamless teamwork within an intelligent agent ecosystem.

OxyGent streamlines the entire lifecycle:

1.  **Code** agents in Python
2.  **Deploy** with a single command
3.  **Monitor** every decision for full transparency
4.  **Evolve** automatically with self-improving systems

## 5. Quick Start

1.  **Set up your environment:**

    ```bash
    conda create -n oxy_env python==3.10
    conda activate oxy_env
    ```

    or

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.10
    uv venv .venv --python 3.10
    source .venv/bin/activate
    ```

2.  **Install OxyGent:**

    ```bash
    pip install oxygent
    ```
    or
    ```bash
    uv pip install oxygent
    ```

3.  **Alternatively set up a developer environment:**

    *   Download **[Node.js](https://nodejs.org)**
    *   Download the requirements:

        ```bash
        pip install -r requirements.txt # or in uv
        brew install coreutils # maybe essential
        ```

4.  **Create a sample Python script (demo.py):**

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

5.  **Configure LLM settings:**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

    or

    ```bash
    # create a .env file
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

6.  **Run the example:**

    ```bash
    python demo.py
    ```

7.  **View the output:**

    ![](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)

## 6. Contributing

Contribute to OxyGent:

1.  Report Issues (Bugs & Errors)
2.  Suggest Enhancements
3.  Improve Documentation
    *   Fork the repository
    *   Add your view in document
    *   Send your pull request
4.  Write Code
    *   Fork the repository
    *   Create a new branch
    *   Add your feature or improvement
    *   Send your pull request

For development help, please see the **[Documentation](http://oxygent.jd.com/docs/)**.

## 7. Community & Support

Submit reproducible steps and log snippets in the project's Issues area or contact the OxyGent Core team.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## 8. About the Contributors

Thank you to the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed.
<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## 9. License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!