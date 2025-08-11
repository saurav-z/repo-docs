# OxyGent: Build Intelligent Systems with Ease 🚀

OxyGent is an open-source Python framework empowering developers to quickly build and deploy production-ready multi-agent AI systems.  [Explore the OxyGent Repository](https://github.com/jd-opensource/OxyGent).

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/jd-opensource/OxyGent/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/jd-opensource/OxyGent/blob/v4/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oxygent.svg?logo=pypi&logoColor=white)](https://pypi.org/project/oxygent/)

<p align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%"/>
</p>

## Key Features

*   **Rapid Development:** Build, deploy, and evolve AI teams efficiently with modular, reusable components.  No more complex YAML configurations, use clean Python interfaces!
*   **Intelligent Collaboration:** Foster dynamic task decomposition, negotiation, and real-time adaptation among agents.
*   **Scalable Architecture:** Support any agent topology and easily optimize performance across distributed systems.
*   **Continuous Evolution:** Leverage built-in evaluation engines for continuous improvement via knowledge feedback loops.
*   **Enterprise-Ready:** Replace siloed AI systems with a unified framework, reducing communication overhead.

## Why OxyGent?

OxyGent provides a complete lifecycle, allowing you to:

1.  **Code** agents in Python.
2.  **Deploy** with a single command.
3.  **Monitor** every decision.
4.  **Evolve** automatically.

## Software Architecture

### Architecture Diagram

  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png" alt="Architecture Diagram" width="100%"/>

### Architecture Components

*   **Repository:** Unified storage for agents, tools, LLMs, data, and system files.
*   **Production Framework:** Comprehensive production chain with registration, building, running, evaluation, and evolution capabilities.
*   **Service Framework:** Robust business system server with complete storage and monitoring support.
*   **Engineering Base:** Extensive external support, including integrated modules like databases and inference engines.

## Quick Start

Get started with OxyGent in just a few steps:

1.  **Set up your environment:**
    *   Using Conda:
        ```bash
        conda create -n oxy_env python==3.10
        conda activate oxy_env
        pip install oxygent
        ```
    *   Using UV:
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv python install 3.10
        uv venv .venv --python 3.10
        source .venv/bin/activate
        uv pip install oxygent
        ```
2.  **Write a sample Python script:**

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

3.  **Configure your LLM settings:**
    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"  # if you want to use a custom base URL
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

    Alternatively, create a `.env` file:

    ```
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```
4.  **Run the example:**

    ```bash
    python demo.py
    ```
5.  **View the output:**

    <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png" alt="Output Example" width="100%"/>

## Contributing

We welcome contributions!  Please follow these steps:

1.  **Fork** the repository.
2.  **Create** a new branch for your feature or improvement.
3.  **Make** your changes.
4.  **Submit** a pull request.

We encourage you to report bugs, suggest enhancements, improve documentation, and contribute code. For more details, check out our document: * **[Document](http://oxygent.jd.com/docs/)**

## Community & Support

If you need help or have questions, please submit reproducible steps and log snippets in the project's Issues area.  You can also reach out to the OxyGent Core team directly via your internal Slack.

<div align="center">
  <img src="https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216" alt="contact" width="50%" height="50%">
</div>

## About the Contributors

Special thanks to all the contributors who have helped build OxyGent!

<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## License

[Apache License](./LICENSE.md)

#### OxyGent is provided by Oxygen JD.com
#### Thanks for your Contributions!
```
Key improvements and reasons:

*   **SEO Optimization:** Added keywords like "Python," "AI," "multi-agent," "framework," and "open-source." Clear headings and subheadings structure the content, improving readability and search engine ranking.
*   **One-Sentence Hook:**  A concise and engaging introductory sentence to immediately grab the reader's attention.
*   **Clearer Organization:**  Uses headings (H2, H3) and bullet points to improve readability.
*   **Concise Summary:** Removed redundant text and focused on the key selling points of OxyGent.
*   **Actionable Quick Start:** Provided more specific, step-by-step instructions for setting up the environment, running the code, and configuring LLM settings. Included both conda and uv install instructions.
*   **Visual Enhancements:** Added image size control for better presentation.
*   **Emphasis on Benefits:** Highlights what OxyGent offers to developers, enterprises, and users.
*   **Contribution Guidance:** Enhanced the contribution section, encouraging community involvement.
*   **Community & Support:** Prominently features ways to get help and contact the developers.
*   **Included the Contact Image**: Kept it as requested.
*   **Removed Irrelevant Information:** Removed the date-specific benchmark information, as it's likely to become outdated quickly.
*   **Corrected Typos and Grammar:** Improved overall clarity.
*   **Comprehensive:** Includes all the essential information from the original README but in a more structured, reader-friendly, and SEO-optimized format.
*   **Removed extraneous HTML**: Replaced with Markdown.
*   **Responsive Images**: Added size constraints to make the images display better on any screen.