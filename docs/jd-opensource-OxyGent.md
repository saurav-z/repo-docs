# OxyGent: Build Advanced Intelligent Systems with Ease

**OxyGent is an open-source Python framework that simplifies the creation of production-ready intelligent systems, enabling developers to build, deploy, and evolve multi-agent systems with unparalleled efficiency.** ([View on GitHub](https://github.com/jd-opensource/OxyGent))

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)
<br/>
<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%" alt="OxyGent Banner">

## Key Features

*   🚀 **Efficient Development:** A modular multi-agent framework with standardized components for rapid agent assembly, hot-swapping, and cross-scenario reuse.
*   🤝 **Intelligent Collaboration:** Dynamic planning paradigms enable agents to decompose tasks, negotiate solutions, and adapt in real-time.
*   🕸️ **Elastic Architecture:** Supports various agent topologies, offering automated dependency mapping and visual debugging tools for optimal performance.
*   🔁 **Continuous Evolution:** Built-in evaluation engines generate training data, enabling agents to continuously improve through knowledge feedback loops.
*   📈 **Scalability:** A distributed scheduler ensures linear cost growth and exponential gains in collaborative intelligence.

## Performance
OxyGent achieved 59.14 points in the GAIA benchmark.

<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png" alt="GAIA points">

## Core Classes
<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="Core Classes">

## Benefits

*   **For Developers:** Focus on business logic without reinventing the wheel.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

## Quick Start

### Step 1: Create and activate a Python environment

*   **Method 1: Conda**

    ```bash
    conda create -n oxy_env python==3.10
    conda activate oxy_env
    ```
*   **Method 2: uv**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.10 
    uv venv .venv --python 3.10
    source .venv/bin/activate
    ```
### Step 2: Install the OxyGent package

*   **Method 1: Conda**

    ```bash
    pip install oxygent
    ```

*   **Method 2: uv**

    ```bash
    uv pip install oxygent
    ```

*   **Method 3: Set develop environment**
    ```bash
    git clone https://github.com/jd-opensource/OxyGent.git
    cd OxyGent
    pip install -r requirements.txt # or in uv
    brew install coreutils # maybe essential
    ```
### Step 3: Install Node.js (if using MCP)

*   Download and install **[Node.js](https://nodejs.org)**

### Step 4: Write a sample Python script (demo.py)

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

    ```
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

### Step 6: Run the example

```bash
python demo.py
```

### Step 7: View the output

<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="Output">

## Contributing

Contribute to OxyGent through:

*   Reporting Issues
*   Suggesting Enhancements
*   Improving Documentation
*   Writing Code

[Contribution Guidelines](http://oxygent.jd.com/docs/)

## Community & Support

For issues, submit reproducible steps and log snippets in the project's [Issues](https://github.com/jd-opensource/OxyGent/issues) area.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Thanks to all [developers](https://github.com/jd-opensource/OxyGent/graphs/contributors).

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License]( ./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!