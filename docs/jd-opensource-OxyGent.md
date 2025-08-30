# OxyGent: Build Production-Ready Intelligent Systems with Ease

**OxyGent** is an open-source Python framework that simplifies the development of multi-agent systems, enabling developers to build advanced, production-ready AI solutions rapidly. ([Original Repository](https://github.com/jd-opensource/OxyGent))

## Key Features

*   🚀 **Rapid Development:** Build and deploy AI teams efficiently with modular components that can be assembled like LEGO bricks.
*   🤝 **Intelligent Collaboration:** Leverage dynamic planning for agent collaboration, task decomposition, and real-time adaptation.
*   🕸️ **Elastic Architecture:** Supports diverse agent topologies, ensuring flexibility and scalability.
*   🔁 **Continuous Evolution:** Agents continuously learn and improve through built-in evaluation engines and knowledge feedback loops.
*   📈 **Scalability:** Utilize a distributed scheduler for linear cost growth and exponential gains in collaborative intelligence.

## Framework Highlights

*   **Focus on Business Logic:**  Developers can concentrate on their core tasks without reinventing the wheel.
*   **Unified Framework for Enterprises:**  Replace siloed AI systems with a cohesive, streamlined framework.
*   **Seamless Teamwork:**  Enable users to experience intelligent interactions through an agent ecosystem.

## Quick Start

Get up and running with OxyGent in a few simple steps:

### 1.  Set Up Your Environment

Choose your preferred method:

*   **Conda:**

    ```bash
    conda create -n oxy_env python==3.10
    conda activate oxy_env
    ```
*   **uv:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.10
    uv venv .venv --python 3.10
    source .venv/bin/activate
    ```

### 2.  Install OxyGent

Choose your preferred method:

*   **Conda:**

    ```bash
    pip install oxygent
    ```
*   **uv:**

    ```bash
    uv pip install oxygent
    ```
*   **Develop Environment:**

    ```bash
    git clone https://github.com/jd-opensource/OxyGent.git
    cd OxyGent
    pip install -r requirements.txt # or in uv
    brew install coreutils # maybe essential
    ```

### 3. (Optional) Install Node.js (for MCP):

*   Download and install **[Node.js](https://nodejs.org)**

### 4. Write a Sample Python Script (demo.py):

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

### 5.  Set Environment Variables

Choose your preferred method:

*   **Terminal:**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```
*   **.env file:**

    ```
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

### 6. Run the Example

```bash
python demo.py
```

### 7. View the Output

(Image of the demo output is here)

## Contributing

We welcome contributions!  You can contribute by:

1.  Reporting issues.
2.  Suggesting enhancements.
3.  Improving documentation.
4.  Writing code.

See the detailed contribution guidelines in the original repository.  For development issues, please check our [Document](http://oxygent.jd.com/docs/)

## Community & Support

Encounter any issues? Please submit reproducible steps and log snippets in the project's Issues area.

Contact the OxyGent Core team:

(Image of the contact is here)

## About the Contributors

A special thanks to all the [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors) who have contributed to OxyGent.

(Image of the contributors is here)

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!