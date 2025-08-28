<!-- Improved README.md for AG2 -->

<a name="readme-top"></a>

<p align="center">
  <img src="https://raw.githubusercontent.com/ag2ai/ag2/27b37494a6f72b1f8050f6bd7be9a7ff232cf749/website/static/img/ag2.svg" width="150" alt="AG2 Logo" title="AG2: Open-Source AgentOS for AI Agents">
  <br>
  <br>

  <!-- Badges -->
  <a href="https://www.pepy.tech/projects/ag2">
    <img src="https://static.pepy.tech/personalized-badge/ag2?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month" alt="Monthly Downloads">
  </a>
  <a href="https://pypi.org/project/autogen/">
    <img src="https://img.shields.io/pypi/v/ag2?label=PyPI&color=green" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/ag2.svg?label=" alt="Python Versions">
  <a href="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml">
    <img src="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml/badge.svg" alt="Build Status">
  </a>
  <a href="https://discord.gg/pAbnFJrkgZ">
    <img src="https://img.shields.io/discord/1153072414184452236?logo=discord&style=flat" alt="Discord">
  </a>
  <a href="https://x.com/ag2oss">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ag2ai" alt="Follow on Twitter">
  </a>

  <br>

  <!-- Links to docs, examples, contributing, etc -->
  <a href="https://docs.ag2.ai/">📚 Documentation</a> |
  <a href="https://github.com/ag2ai/build-with-ag2">💡 Examples</a> |
  <a href="https://docs.ag2.ai/latest/docs/contributor-guide/contributing">🤝 Contributing</a> |
  <a href="#related-papers">📝 Cite Paper</a> |
  <a href="https://discord.gg/pAbnFJrkgZ">💬 Join Discord</a>
</p>

<p align="center">
  AG2, evolved from AutoGen, is your gateway to building powerful AI agents and multi-agent systems.
</p>

# AG2: Revolutionizing AI Agent Development with Open-Source Technology

AG2 ([GitHub](https://github.com/ag2ai/ag2)), formerly known as AutoGen, is an open-source framework designed to simplify the development and research of AI agents and multi-agent systems. Build AI agents that interact, collaborate, and solve complex tasks efficiently, leveraging the power of large language models (LLMs) and tool integrations.  Join a vibrant community of developers and researchers to shape the future of agentic AI.

## Key Features

*   **Multi-Agent Collaboration:** Design and orchestrate complex workflows with agents that can communicate and cooperate.
*   **LLM & Tool Integration:** Seamlessly incorporate various LLMs and external tools for enhanced capabilities.
*   **Flexible Workflows:** Support for autonomous and human-in-the-loop agent interactions.
*   **Conversation Patterns:** Built-in patterns such as swarms, group chats, and custom orchestration.
*   **Open-Source & Community Driven:** Benefit from a collaborative environment with a focus on open governance.

## Table of Contents

*   [AG2: Revolutionizing AI Agent Development with Open-Source Technology](#ag2-revolutionizing-ai-agent-development-with-open-source-technology)
    *   [Key Features](#key-features)
    *   [Table of Contents](#table-of-contents)
    *   [Getting Started](#getting-started)
        *   [Installation](#installation)
        *   [Setup API Keys](#setup-api-keys)
        *   [Run Your First Agent](#run-your-first-agent)
    *   [Example Applications](#example-applications)
    *   [Agent Concepts](#agent-concepts)
        *   [Conversable Agent](#conversable-agent)
        *   [Human-in-the-Loop](#human-in-the-loop)
        *   [Orchestrating Multiple Agents](#orchestrating-multiple-agents)
        *   [Tools](#tools)
        *   [Advanced Agentic Design Patterns](#advanced-agentic-design-patterns)
    *   [Announcements](#announcements)
    *   [Contributors Wall](#contributors-wall)
    *   [Code Style and Linting](#code-style-and-linting)
    *   [Related Papers](#related-papers)
    *   [Cite the Project](#cite-the-project)
    *   [License](#license)

## Getting Started

Kickstart your AG2 journey with the following steps and explore the [Basic Concepts](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/installing-ag2/) in our documentation for a detailed walkthrough.

### Installation

Ensure you have Python version >= 3.10, < 3.14. Install AG2 using pip:

```bash
pip install ag2[openai]
```

Install extra options based on the features you need.

### Setup API Keys

To keep your LLM dependencies neat, we recommend using the `OAI_CONFIG_LIST` file to store your API keys.

Use the sample file `OAI_CONFIG_LIST_sample` as a template.

```json
[
  {
    "model": "gpt-4o",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run Your First Agent

Create a Python script or Jupyter Notebook to run your first agent:

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")


with llm_config:
    assistant = AssistantAgent("assistant")
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
# This initiates an automated chat between the two agents to solve the task
```

## Example Applications

Explore a range of example applications to help you get started, or check out our collection of Jupyter notebooks.

*   [Build with AG2](https://github.com/ag2ai/build-with-ag2)
*   [Jupyter Notebooks](notebook)

## Agent Concepts

AG2 introduces core agent concepts to help you build powerful AI agents.

### Conversable Agent

The `ConversableAgent` ([documentation](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent)) is the foundation for communication between AI entities.

Example:

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

with llm_config:
  assistant = ConversableAgent(
      name="assistant",
      system_message="You are an assistant that responds concisely.",
  )

  fact_checker = ConversableAgent(
      name="fact_checker",
      system_message="You are a fact-checking assistant.",
  )

assistant.initiate_chat(
    recipient=fact_checker,
    message="What is AG2?",
    max_turns=2
)
```

### Human-in-the-Loop

Enhance AI workflows with human oversight using the `human_input_mode` parameter.

```python
from autogen import ConversableAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

with llm_config:
  assistant = ConversableAgent(
      name="assistant",
      system_message="You are a helpful assistant.",
  )

human = ConversableAgent(
    name="human",
    human_input_mode="ALWAYS"
)

human.initiate_chat(
    recipient=assistant,
    message="Hello! What's 2 + 2?"
)
```

### Orchestrating Multiple Agents

Create collaborative systems with patterns like `GroupChat` and `Swarm`.

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

planner_message = """You are a classroom lesson agent.
Given a topic, write a lesson plan for a fourth grade class.
Use the following format:
<title>Lesson plan title</title>
<learning_objectives>Key learning objectives</learning_objectives>
<script>How to introduce the topic to the kids</script>
"""

reviewer_message = """You are a classroom lesson reviewer.
You compare the lesson plan to the fourth grade curriculum and provide a maximum of 3 recommended changes.
Provide only one round of reviews to a lesson plan.
"""

with llm_config:
    lesson_planner = ConversableAgent(
        name="planner_agent",
        system_message=planner_message,
    )

    lesson_reviewer = ConversableAgent(
        name="reviewer_agent",
        system_message=reviewer_message,
    )

with llm_config:
    teacher = ConversableAgent(
        name="teacher_agent",
        system_message="You are a classroom teacher.",
        is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),
    )

groupchat = GroupChat(
    agents=[teacher, lesson_planner, lesson_reviewer],
    speaker_selection_method="auto",
    messages=[],
)

manager = GroupChatManager(
    name="group_manager",
    groupchat=groupchat,
    llm_config=llm_config,
)

teacher.initiate_chat(
    recipient=manager,
    message="Today, let's introduce our kids to the solar system."
)
```

### Tools

Equip agents with tools for external data access and functionality.

```python
from datetime import datetime
from typing import Annotated

from autogen import ConversableAgent, register_function, LLMConfig

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")

with llm_config:
    date_agent = ConversableAgent(
        name="date_agent",
        system_message="You get the day of the week for a given date.",
    )

executor_agent = ConversableAgent(
    name="executor_agent",
    human_input_mode="NEVER",
)

register_function(
    get_weekday,
    caller=date_agent,
    executor=executor_agent,
    description="Get the day of the week for a given date",
)

chat_result = executor_agent.initiate_chat(
    recipient=date_agent,
    message="I was born on the 25th of March 1995, what day was it?",
    max_turns=2,
)

print(chat_result.chat_history[-1]["content"])
```

### Advanced Agentic Design Patterns

Explore advanced concepts, including:

*   [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
*   [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
*   [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
*   [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
*   [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)

## Announcements

*   🔥 🎉 **Nov 11, 2024:** AutoGen evolves into **AG2**! [AG2AI](https://github.com/ag2ai) is now hosting development. Check out [AG2's new look](https://ag2.ai/).
*   📄 **License:** Apache 2.0 license from v0.3.
*   🎉 May 29, 2024: DeepLearning.ai launched a new short course [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen), made in collaboration with Microsoft and Penn State University, and taught by AutoGen creators [Chi Wang](https://github.com/sonichi) and [Qingyun Wu](https://github.com/qingyun-wu).
*   🎉 May 24, 2024: Foundation Capital published an article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and a video [AI in the Real World Episode 2: Exploring Multi-Agent AI and AutoGen with Chi Wang](https://www.youtube.com/watch?v=RLwyXRVvlNk).
*   🎉 Apr 17, 2024: Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and [What's next for AI agentic workflows](https://youtu.be/sal78ACtGTc?si=JduUzN_1kDnMq0vF) at Sequoia Capital's AI Ascent (Mar 26).
*   [More Announcements](announcements.md)

## Contributors Wall

<!-- Contributor Image -->
<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" alt="Contributors"/>
</a>

## Code Style and Linting

This project uses pre-commit hooks for code quality. Ensure you have:

1.  Install pre-commit:

    ```bash
    pip install pre-commit
    pre-commit install
    ```

2.  Run hooks manually:

    ```bash
    pre-commit run --all-files
    ```

## Related Papers

*   [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
*   [EcoOptiGen: Hyperparameter Optimization for Large Language Model Generation Inference](https://arxiv.org/abs/2303.04673)
*   [MathChat: Converse to Tackle Challenging Math Problems with LLM Agents](https://arxiv.org/abs/2306.01337)
*   [AgentOptimizer: Offline Training of Language Model Agents with Functions as Learnable Weights](https://arxiv.org/pdf/2402.11359)
*   [StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows](https://arxiv.org/abs/2403.11322)

## Cite the Project

```
@software{AG2_2024,
author = {Chi Wang and Qingyun Wu and the AG2 Community},
title = {AG2: Open-Source AgentOS for AI Agents},
year = {2024},
url = {https://github.com/ag2ai/ag2},
note = {Available at https://docs.ag2.ai/},
version = {latest}
}
```

## License

This project is licensed under the [Apache License, Version 2.0 (Apache-2.0)](./LICENSE).

This project is a spin-off of [AutoGen](https://github.com/microsoft/autogen) and contains code under two licenses:

*   The original code from https://github.com/microsoft/autogen is licensed under the MIT License. See the [LICENSE\_original\_MIT](./license_original/LICENSE\_original\_MIT) file for details.

*   Modifications and additions made in this fork are licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for the full license text.

We have documented these changes for clarity and to ensure transparency with our user and contributor community. For more details, please see the [NOTICE](./NOTICE.md) file.