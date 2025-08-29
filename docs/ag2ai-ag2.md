<a name="readme-top"></a>

<p align="center">
  <!-- The image URL points to the GitHub-hosted content, ensuring it displays correctly on the PyPI website.-->
  <img src="https://raw.githubusercontent.com/ag2ai/ag2/27b37494a6f72b1f8050f6bd7be9a7ff232cf749/website/static/img/ag2.svg" width="150" title="AG2 Logo">

  <br>
  <br>

  <a href="https://www.pepy.tech/projects/ag2">
    <img src="https://static.pepy.tech/personalized-badge/ag2?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month" alt="Downloads"/>
  </a>

  <a href="https://pypi.org/project/autogen/">
    <img src="https://img.shields.io/pypi/v/ag2?label=PyPI&color=green">
  </a>

  <img src="https://img.shields.io/pypi/pyversions/ag2.svg?label=">

  <a href="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml">
    <img src="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml/badge.svg">
  </a>
  <a href="https://discord.gg/pAbnFJrkgZ">
    <img src="https://img.shields.io/discord/1153072414184452236?logo=discord&style=flat">
  </a>

  <br>

  <a href="https://x.com/ag2oss">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ag2ai">
  </a>
</p>

<p align="center">
  <a href="https://docs.ag2.ai/">📚 Documentation</a> |
  <a href="https://github.com/ag2ai/build-with-ag2">💡 Examples</a> |
  <a href="https://docs.ag2.ai/latest/docs/contributor-guide/contributing">🤝 Contributing</a> |
  <a href="#related-papers">📝 Cite paper</a> |
  <a href="https://discord.gg/pAbnFJrkgZ">💬 Join Discord</a>
</p>

<p align="center">
  AG2, evolved from AutoGen, empowers you to build and orchestrate AI agents for cutting-edge applications.
</p>

# AG2: The Open-Source AgentOS for AI Agents

AG2 (formerly AutoGen) is a powerful, open-source framework designed for building and managing AI agents.  It enables developers to create sophisticated multi-agent systems that can collaborate to solve complex tasks, leveraging the power of Large Language Models (LLMs) and various tools.  AG2 streamlines the development and research of agentic AI, offering features like:

*   **Multi-Agent Collaboration:**  Facilitate seamless interaction and cooperation between multiple AI agents.
*   **LLM & Tool Integration:** Support for a wide range of LLMs and tool usage within your agent workflows.
*   **Flexible Workflows:**  Enable both autonomous and human-in-the-loop workflows.
*   **Conversation Patterns:**  Implement advanced multi-agent conversation patterns.

**[Explore the AG2 Repository on GitHub](https://github.com/ag2ai/ag2)**

AG2 is actively maintained by a community of dedicated volunteers. For maintainer inquiries, contact Chi Wang and Qingyun Wu at [support@ag2.ai](mailto:support@ag2.ai).

## Key Features

*   **Conversable Agents:**  Build agents capable of engaging in conversations, receiving input, and generating responses.
*   **Human-in-the-Loop:** Easily integrate human oversight and feedback into your agent workflows.
*   **Agent Orchestration:** Design and manage complex multi-agent interactions using built-in patterns or custom designs.
*   **Tool Integration:** Empower agents with access to external tools, APIs, and data.
*   **Advanced Design Patterns:** Utilize features like structured outputs, RAG, code execution and more.

## Table of Contents

*   [Key Features](#key-features)
*   [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Setup your API keys](#setup-your-api-keys)
    *   [Run your first agent](#run-your-first-agent)
*   [Example Applications](#example-applications)
*   [Agent Concepts](#introduction-of-different-agent-concepts)
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

Dive into AG2 with a step-by-step introduction to key concepts and code examples. [Basic Concepts](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/installing-ag2/) in our documentation.

### Installation

Ensure you have **Python version >= 3.10, < 3.14** installed. Install AG2 via pip:

```bash
pip install ag2[openai]
```

Install optional dependencies as needed.

### Setup your API Keys

Manage your API keys effectively using the `OAI_CONFIG_LIST` file.  Use `OAI_CONFIG_LIST_sample` as a template.

```json
[
  {
    "model": "gpt-4o",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run your first agent

Here's a simple example:

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

Explore ready-to-use applications to get started or check out our collection of Jupyter notebooks.

*   [Build with AG2](https://github.com/ag2ai/build-with-ag2)
*   [Jupyter Notebooks](notebook)

## Agent Concepts

AG2 provides core agent concepts to help you build your AI agents.

### Conversable Agent

The [ConversableAgent](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent) is the fundamental building block, enabling seamless communication.

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

Easily integrate human input for critical decisions.

```python
from autogen import ConversableAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

with llm_config:
  assistant = ConversableAgent(
      name="assistant",
      system_message="You are a helpful assistant.",
  )
human = UserProxyAgent(name="human", code_execution_config={"work_dir": "coding", "use_docker": False})
human.initiate_chat(
    recipient=assistant,
    message="Hello! What's 2 + 2?"
)
```

### Orchestrating Multiple Agents

Create sophisticated multi-agent systems with flexible patterns.

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

planner_message = """...lesson plan content..."""
reviewer_message = """...review content..."""

with llm_config:
    lesson_planner = ConversableAgent(...)
    lesson_reviewer = ConversableAgent(...)

with llm_config:
    teacher = ConversableAgent(...)

groupchat = GroupChat(...)
manager = GroupChatManager(...)

teacher.initiate_chat(
    recipient=manager,
    message="Today, let's introduce our kids to the solar system."
)
```

### Tools

Empower agents with tools for external access.

```python
from datetime import datetime
from typing import Annotated
from autogen import ConversableAgent, register_function, LLMConfig

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")

with llm_config:
    date_agent = ConversableAgent(...)
executor_agent = ConversableAgent(...)

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

Explore advanced concepts.

*   [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
*   [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
*   [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
*   [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
*   [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)

## Announcements

*   🔥 **Nov 11, 2024:** AutoGen is now **AG2**!  [AG2AI](https://github.com/ag2ai) is the new home for AG2. Check out [AG2's new look](https://ag2.ai/).
*   📄 **License:** Apache 2.0 from v0.3.
*   🎉 May 29, 2024: DeepLearning.ai [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen) course.
*   🎉 May 24, 2024: Foundation Capital article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and video.
*   🎉 Apr 17, 2024: Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and at Sequoia Capital's AI Ascent (Mar 26).

[More Announcements](announcements.md)

## Contributors Wall

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

## Code Style and Linting

Maintain code quality with pre-commit hooks.

1.  Install: `pip install pre-commit`
2.  Install hooks: `pre-commit install`
3.  Run manually: `pre-commit run --all-files`

## Related Papers

*   [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
*   ... (and others)

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

It is a spin-off of [AutoGen](https://github.com/microsoft/autogen) and includes code under two licenses:

*   Original code from https://github.com/microsoft/autogen is licensed under the MIT License. See [LICENSE_original_MIT](./license_original/LICENSE_original_MIT).
*   Modifications and additions are licensed under the Apache License, Version 2.0. See [LICENSE](./LICENSE).
  For more details, please see the [NOTICE](./NOTICE.md) file.