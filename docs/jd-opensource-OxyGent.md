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

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<h1 align="center">OxyGent: Build Intelligent Systems Faster</h1>

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%">
</p>

<h3 align="center">
  OxyGent is an advanced Python framework that empowers developers to rapidly build production-ready intelligent systems.
  <br>
  Visit our website: <a href="http://oxygent.jd.com">OxyGent</a>
  <br>
  <a href="https://github.com/jd-opensource/OxyGent">View the Original Repository</a>
</h3>

## Key Features of OxyGent

*   **Modular Multi-Agent Framework:** Build, deploy, and evolve AI teams with unparalleled efficiency using modular components.
*   **Intelligent Collaboration:** Facilitates dynamic planning, task decomposition, and negotiation among agents for robust solutions.
*   **Elastic Architecture:** Supports diverse agent topologies, providing flexibility and scalability.
*   **Continuous Evolution:** Leverages built-in evaluation engines and feedback loops to continuously improve agent performance.
*   **Scalability:** Employs a distributed scheduler for linear cost growth and exponential gains in collaborative intelligence.

## 1. Project Overview

**OxyGent** is an open-source framework designed to unify tools, models, and agents into modular "Oxy" components. This framework empowers developers to create end-to-end AI pipelines with ease, facilitating the seamless building, running, and evolution of multi-agent systems.

## 2. Core Features - Deep Dive

*   **Efficient Development:** Assemble agents quickly with standardized Oxy components. Supports hot-swapping and cross-scenario reuse through clean Python interfaces.
*   **Intelligent Collaboration:** Agents intelligently decompose tasks, negotiate, and adapt in real-time, ensuring full auditability.
*   **Elastic Architecture:** Supports a variety of agent topologies (e.g., ReAct, hybrid planning). Automated dependency mapping and debugging tools streamline performance optimization.
*   **Continuous Evolution:** Evaluation engines auto-generate training data, allowing agents to learn from interactions.
*   **Scalability:**  The distributed scheduler enables domain-wide optimization and real-time decision-making at any scale.

**Performance:** In the July 15, 2025, GAIA benchmark, OxyGent achieved 59.14 points, closely approaching the top open-source system, OWL, which scored 60.8 points.

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png" alt="OxyGent Performance" width="70%">
</p>

## 3. Software Architecture

### 3.1 Diagram

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="OxyGent Architecture" width="100%">
</p>

### 3.2 Architecture Description

*   **Repository:** Unified storage for agents, tools, LLMs, data, and system files.
*   **Production Framework:** Complete production chain, including registration, building, running, evaluation, and evolution functionalities.
*   **Service Framework:** Complete business system server providing comprehensive storage and monitoring support.
*   **Engineering Base:** Rich external support, including integrated modules such as databases and inference engines.

## 4. Feature Highlights

**OxyGent benefits:**

*   **For Developers:** Focus on business logic without the need to reinvent the wheel.
*   **For Enterprises:** Unify siloed AI systems within a single framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

**Lifecycle:**

1.  **Code** agents in Python.
2.  **Deploy** with a single command.
3.  **Monitor** every decision for full transparency.
4.  **Evolve** systems automatically.

## 5. Quick Start

Get started with OxyGent in these easy steps:

1.  **Set Up Your Environment:**

    *   **Conda:**
        ```bash
        conda create -n oxy_env python==3.10
        conda activate oxy_env
        ```

    *   **UV:**
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv python install 3.10
        uv venv .venv --python 3.10
        source .venv/bin/activate
        ```

2.  **Install OxyGent:**

    *   **Conda:**
        ```bash
        pip install oxygent
        ```
    *   **UV:**
        ```bash
        uv pip install oxygent
        ```

3.  **Configure the Development Environment (Optional):**

    *   Install **[Node.js](https://nodejs.org)**.
    *   Install required dependencies:
        ```bash
        pip install -r requirements.txt # or in uv
        brew install coreutils # maybe essential
        ```

4.  **Write a Sample Python Script:**
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

5.  **Configure LLM Settings:**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```
    Or create a `.env` file:
    ```bash
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

6.  **Run the Example:**
    ```bash
    python demo.py
    ```

7.  **View the Output:**
    <p align="center">
      <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="OxyGent Output" width="70%">
    </p>

## 6. Contributing

Contribute to OxyGent and help improve this powerful framework.

**Ways to Contribute:**

1.  Report Issues (Bugs & Errors).
2.  Suggest Enhancements.
3.  Improve Documentation:
    *   Fork the repository.
    *   Add your view in document.
    *   Send your pull request.
4.  Write Code:
    *   Fork the repository.
    *   Create a new branch.
    *   Add your feature or improvement.
    *   Send your pull request.

For development-related issues, please consult the documentation: **[Document](http://oxygent.jd.com/docs/)**

## 7. Community & Support

Find help and support for OxyGent:

*   Submit reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area.
*   Contact the OxyGent Core team via your internal Slack (if applicable).

<p align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="Contact Us" width="50%">
</p>

## 8. About the Contributors

We appreciate all contributions! 🎉🎉🎉

Thanks to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.
<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## 9. License

[Apache License](./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!