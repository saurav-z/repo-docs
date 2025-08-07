# OxyGent: Build Intelligent Systems Faster with this Open-Source Framework

**OxyGent** empowers developers to rapidly build and deploy production-ready intelligent systems, streamlining development and accelerating innovation. [Explore the OxyGent repository](https://github.com/jd-opensource/OxyGent)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%" alt="OxyGent Banner">
</p>

<p align="center">
  Visit our website:
  <a href="http://oxygent.jd.com">OxyGent Website</a>
</p>

## Key Features of OxyGent

*   **🚀 Efficient Development:** Build, deploy, and evolve AI teams with unprecedented speed using modular, reusable components.

*   **🤝 Intelligent Collaboration:**  Facilitate dynamic task decomposition, negotiation, and real-time adaptation for robust multi-agent systems.

*   **🕸️ Elastic Architecture:** Supports diverse agent topologies and provides tools for performance optimization across distributed systems.

*   **🔁 Continuous Evolution:**  Leverage built-in evaluation engines and knowledge feedback loops for continuous agent improvement and full transparency.

*   **📈 Scalability:**  Achieve linear cost growth while delivering exponential gains in collaborative intelligence with OxyGent's distributed scheduler.

## 1. Project Overview

OxyGent is an open-source framework designed to unify tools, models, and agents, fostering the development of sophisticated multi-agent systems. It offers a streamlined, end-to-end pipeline that simplifies building, running, and evolving AI-powered applications.

## 2. Core Features in Detail

*   **Modular Components:** OxyGent offers standardized components for easy agent assembly, supporting hot-swapping and cross-scenario reuse.
*   **Dynamic Planning:** Agents intelligently decompose tasks, negotiate, and adapt in real-time.
*   **Auditability:** Maintain complete transparency of every agent decision.
*   **Flexible Architecture:** Supports various agent topologies.
*   **Automated Optimization:** Utilize automated dependency mapping and visual debugging tools for performance enhancement.
*   **Evaluation Engines:** Built-in evaluation engines auto-generate training data.
*   **Scalable Design:** OxyGent's distributed scheduler enables scalable collaborative intelligence.

## 3. Software Architecture

### 3.1 Diagram

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="OxyGent Architecture Diagram">
</p>

### 3.2 Architecture Description

*   **📦 Repository:** Centralized storage for agents, tools, LLMs, data, and system files.
*   **🛠 Production Framework:** Includes registration, building, running, evaluation, and evolution.
*   **🖥 Service Framework:** Provides a comprehensive business system server with storage and monitoring.
*   **⚙️ Engineering Base:** Offers extensive external support, including databases and inference engines.

## 4. Feature Highlights

*   **For Developers:**  Focus on core business logic without reinventing the wheel.
*   **For Enterprises:**  Replace siloed AI systems with a unified framework, streamlining communication.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

OxyGent streamlines the entire lifecycle:

1.  **Code** agents in Python (no YAML needed).
2.  **Deploy** with a single command (local or cloud).
3.  **Monitor** every decision (full transparency).
4.  **Evolve** automatically (self-improving systems).

## 5. Quick Start

### Prerequisites

*   Python 3.10 or higher
*   [Node.js](https://nodejs.org) (for development environment)

### Installation

**Using Conda:**

```bash
conda create -n oxy_env python==3.10
conda activate oxy_env
pip install oxygent
```

**Using uv:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install oxygent
```

### Development Environment (Optional)

```bash
pip install -r requirements.txt # or in uv
brew install coreutils # maybe essential
```

### Example Usage

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

### Configure LLM settings

```bash
export DEFAULT_LLM_API_KEY="your_api_key"
export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
export DEFAULT_LLM_MODEL_NAME="your_model_name"
```

**Or using a .env file:**

```bash
# create a .env file
DEFAULT_LLM_API_KEY="your_api_key"
DEFAULT_LLM_BASE_URL="your_base_url"
DEFAULT_LLM_MODEL_NAME="your_model_name"
```

### Run the Example

```bash
python demo.py
```

### Output

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="Example Output">
</p>

## 6. Contributing

Contributions to OxyGent are highly welcomed! You can contribute by:

1.  Reporting Issues (Bugs & Errors)
2.  Suggesting Enhancements
3.  Improving Documentation
    *   Fork the repository
    *   Add your views in the document
    *   Send your pull request
4.  Writing Code
    *   Fork the repository
    *   Create a new branch
    *   Add your feature or improvement
    *   Send your pull request

Refer to the [OxyGent documentation](http://oxygent.jd.com/docs/) for more information.

## 7. Community & Support

For assistance or to report issues, please submit reproducible steps and log snippets in the project's Issues area or contact the OxyGent Core team directly.

Contact Us:

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## 8. About the Contributors

Thank you to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## 9. License

[Apache License](./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!