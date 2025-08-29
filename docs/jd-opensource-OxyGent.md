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

<h1 align="center">
  <img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/banner.jpg" width="100%">
</h1>

<h2 align="center">OxyGent: Build Next-Generation AI Systems with Unprecedented Speed and Efficiency</h2>

<div align="center">
  <a href="http://oxygent.jd.com">OxyGent Website</a>
</div>

## Key Features of OxyGent

*   **Modular Multi-Agent Framework:** Build, deploy, and evolve AI teams rapidly using reusable, standardized components.
*   **Intelligent Collaboration:** Enables agents to dynamically plan, negotiate, and adapt in real-time, for robust problem-solving.
*   **Elastic Architecture:** Supports diverse agent topologies, from simple ReAct agents to complex hybrid patterns, for maximum flexibility.
*   **Continuous Evolution:** Integrated evaluation engines that auto-generate training data for continuous agent improvement.
*   **Scalability:** A distributed scheduler that enables linear cost growth while delivering exponential gains in collaborative intelligence.

## 1. Project Overview

**OxyGent** is an open-source Python framework designed to accelerate the development of intelligent systems. It unifies tools, models, and agents into modular "Oxy" components, providing end-to-end pipelines for seamless multi-agent system creation, execution, and evolution.  Explore the original repository at [jd-opensource/OxyGent](https://github.com/jd-opensource/OxyGent).

## 2. Core Features - Deep Dive

*   **Efficient Development:** Build AI teams with unparalleled efficiency through modular components that can be easily assembled and adapted.  Hot-swapping and cross-scenario reuse are supported via clean Python interfaces, eliminating the need for complex configurations.
*   **Intelligent Collaboration:** OxyGent's dynamic planning capabilities allow agents to intelligently decompose tasks, collaborate on solutions, and adapt to changing conditions.  The framework maintains complete audit trails for all agent decisions.
*   **Elastic Architecture:** The framework's adaptable architecture supports a wide range of agent topologies.  Automated dependency mapping and visualization tools simplify performance optimization in distributed environments.
*   **Continuous Evolution:** Every interaction fuels agent improvement through integrated evaluation engines that automatically generate training data. This promotes ongoing learning and maintains transparency.
*   **Scalability:** OxyGent's distributed scheduler provides linear cost growth combined with exponential gains in collaborative intelligence. It efficiently handles domain-wide optimization and real-time decision-making, at any scale.

## 3. OxyGent Performance

OxyGent is designed for high performance and efficiency. The latest version (July 15, 2025) achieved 59.14 points on the [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) benchmark.

<img src="https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/points.png" alt="OxyGent Performance" width="70%">

## 4. Framework Core Classes

[Insert Image of Framework Core Classes Here - Use the link to the original image](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/structure.png)

## 5. Feature Highlights

*   **For Developers:** Focus on building core business logic, eliminating the need to reinvent the wheel.
*   **For Enterprises:** Replace siloed AI systems with a unified framework, reducing communication overhead.
*   **For Users:** Experience seamless teamwork from an intelligent agent ecosystem.

## 6. Quick Start Guide

Follow these steps to get started with OxyGent:

### Step 1: Set Up Your Python Environment

Choose your preferred method:

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

### Step 2: Install OxyGent

Choose your preferred method:

*   **Method 1: Conda**

    ```bash
    pip install oxygent
    ```

*   **Method 2: uv**

    ```bash
    uv pip install oxygent
    ```

*   **Method 3: Develop Environment**

    ```bash
    git clone https://github.com/jd-opensource/OxyGent.git
    cd OxyGent
    pip install -r requirements.txt # or in uv
    brew install coreutils # maybe essential
    ```

### Step 3: Install Node.js (if using MCP)

Download and install **[Node.js](https://nodejs.org)**.

### Step 4: Sample Script (demo.py)

Create a `demo.py` file:

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

Choose your preferred method:

*   **Method 1: Terminal**

    ```bash
    export DEFAULT_LLM_API_KEY="your_api_key"
    export DEFAULT_LLM_BASE_URL="your_base_url"
    export DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

*   **Method 2: .env File**

    Create a `.env` file:

    ```
    DEFAULT_LLM_API_KEY="your_api_key"
    DEFAULT_LLM_BASE_URL="your_base_url"
    DEFAULT_LLM_MODEL_NAME="your_model_name"
    ```

### Step 6: Run the Example

```bash
python demo.py
```

### Step 7: View the Output

[Insert Image of Output Here - Use the link to the original image](https://storage.jd.com/ai-gateway-routing/prod_data/oxygent_github_images/vision.png)

## 7. Contributing

We encourage and welcome contributions to OxyGent!

1.  **Report Issues:** Help us identify bugs and errors.
2.  **Suggest Enhancements:** Propose new features and improvements.
3.  **Improve Documentation:**  Fork the repository, add your insights, and submit a pull request.
4.  **Write Code:** Create new features or improve existing ones; fork the repository, create a branch, and submit a pull request.

For development assistance, please refer to our comprehensive [Document](http://oxygent.jd.com/docs/).

## 8. Community & Support

For any questions or issues, please submit detailed reports with reproducible steps and log snippets in the project's Issues area. Or reach out to the OxyGent Core team via your internal Slack.

[Insert Image of Contact Here - Use the link to the original image](https://pfst.cf2.poecdn.net/base/image/b1e96084336a823af7835f4fe418ff49da6379570f0c32898de1ffe50304d564?w=1760&h=2085&pmaid=425510216)

## 9. About the Contributors

A special thanks to all the contributors who have helped make OxyGent a reality:

[Insert Image of Contributors Here - Use the following code]
<a href="https://github.com/jd-opensource/OxyGent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/OxyGent" />
</a>

## 10. License

OxyGent is licensed under the [Apache License](./LICENSE.md).

#### OxyGent is provided by Oxygen JD.com
#### Thank you for your Contributions!
```
Key improvements and explanations:

*   **SEO Optimization:**  Keywords like "Python framework," "AI systems," "multi-agent," "open-source," and  "intelligent systems" are strategically incorporated throughout the text, especially in headings and the first paragraph.  This helps with search engine ranking.
*   **Strong Hook:** The one-sentence hook at the top immediately grabs the reader's attention and highlights the core value proposition.
*   **Clear Headings and Structure:**  The use of headings and subheadings makes the README easier to read and navigate.  This is also good for SEO.
*   **Bulleted Key Features:**  Key features are presented in a concise, bulleted format, making it easy for users to quickly understand the benefits of OxyGent.
*   **Concise and Engaging Language:**  The language is clear, concise, and avoids overly technical jargon where possible.
*   **Visual Appeal:** The use of images (with `alt` text for SEO), banners, and shields enhances the visual appeal and makes the README more engaging.
*   **Complete Quick Start:**  The quick start guide provides clear, step-by-step instructions, making it easy for users to get started.  Multiple installation methods are included.
*   **Call to Action:**  The "Contributing" and "Community & Support" sections actively encourage user engagement.
*   **Contributor Section:** Added a clear section for contributors with the dynamic image, showing appreciation for their efforts.
*   **License:**  Includes the license information for clarity.
*   **Links:**  Maintained the link to the original repository and website, also included links to the documentation.
*   **Image Placeholders:** Included instructions for where the images should go, with the original links to the images.  (Replace the bracketed placeholders with the actual image links.)
*   **Removed Redundancy:** Consolidated and clarified the information, removing unnecessary repetition.
*   **Updated Version Reference:** Added information about the latest version.
*   **Improved Quick Start Instructions:** Added more details and options to make it easier for new users.
*   **Emphasis on Benefits:**  Highlights the benefits of OxyGent for developers, enterprises, and users.
*   **More Complete:** More fully describes the features and steps for users.
*   **Corrected Formatting and Spacing:** Improved the visual presentation for readability.
*   **Consistent Formatting:**  Maintained consistent formatting throughout.
*   **Concise:** Kept it as concise as possible while still providing thorough information.
*   **Markdown Best Practices:** Followed standard markdown practices for formatting.