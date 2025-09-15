<!-- Improved README - SEO Optimized -->
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
  AG2 was evolved from AutoGen. Fully open-sourced. We invite collaborators from all organizations to contribute.
</p>

# AG2: Build Powerful AI Agents with Ease

**AG2 is a cutting-edge, open-source framework empowering developers to create and orchestrate AI agents for complex tasks.** This document provides an overview of AG2, highlighting its key features, usage, and resources.

## Key Features of AG2

*   **Multi-Agent Collaboration:** Design agents that interact with each other, facilitating complex task solving.
*   **LLM Agnostic:** Seamlessly integrate with various Large Language Models (LLMs).
*   **Tool Integration:** Empower agents with the ability to use tools and external APIs.
*   **Human-in-the-Loop Workflows:** Incorporate human oversight and feedback for enhanced control.
*   **Flexible Orchestration:** Leverage built-in conversation patterns (swarms, group chats) or customize agent interactions.

## Table of Contents

-   [AG2: Build Powerful AI Agents with Ease](#ag2-build-powerful-ai-agents-with-ease)
    -   [Key Features of AG2](#key-features-of-ag2)
    -   [Table of Contents](#table-of-contents)
    -   [Getting Started](#getting-started)
        -   [Installation](#installation)
        -   [Setting Up API Keys](#setup-your-api-keys)
        -   [Run Your First Agent](#run-your-first-agent)
    -   [Example Applications](#example-applications)
    -   [Core Agent Concepts](#introduction-of-different-agent-concepts)
        -   [Conversable Agent](#conversable-agent)
        -   [Human-in-the-Loop](#human-in-the-loop)
        -   [Orchestrating Multiple Agents](#orchestrating-multiple-agents)
        -   [Tools](#tools)
        -   [Advanced Agentic Design Patterns](#advanced-agentic-design-patterns)
    -   [Announcements](#announcements)
    -   [Contributors Wall](#contributors-wall)
    -   [Code Style and Linting](#code-style-and-linting)
    -   [Related Papers](#related-papers)
    -   [Cite the Project](#cite-the-project)
    -   [License](#license)

## Getting Started

This section provides a quick start guide to using AG2. For more detailed information and tutorials, consult the [Basic Concepts](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/installing-ag2/) section in our documentation.

### Installation

AG2 requires **Python version >= 3.10, < 3.14**. Install AG2 using pip:

```bash
pip install ag2[openai]
```

Install optional dependencies for specific features:

```bash
# Example: install dependencies for a specific feature
pip install ag2[<optional_feature_name>]
```

### Setting Up API Keys

Manage your LLM dependencies efficiently using the `OAI_CONFIG_LIST` file. Use `OAI_CONFIG_LIST_sample` as a template.

```json
[
  {
    "model": "gpt-5",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run Your First Agent

Create a Python script or Jupyter Notebook and run your first agent.

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

assistant = AssistantAgent("assistant", llm_config=llm_config)

user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})

user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
# This initiates an automated chat between the two agents to solve the task
```

## Example Applications

Explore various use cases and get inspired by our comprehensive range of applications:

*   **Build with AG2 Examples Repository:** [Build with AG2](https://github.com/ag2ai/build-with-ag2)
*   **Jupyter Notebooks:** [Jupyter Notebooks](notebook)

## Core Agent Concepts

AG2 introduces fundamental agent concepts to build robust AI systems:

### Conversable Agent

The [ConversableAgent](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent) is the foundation for agent communication, designed for seamless message exchange and response generation.

```python
# Example ConversableAgent usage
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig({
    "api_type": "openai",
    "model": "gpt-5-mini",
})

assistant = ConversableAgent(
    name="assistant",
    system_message="You are an assistant that responds concisely.",
    llm_config=llm_config,
)

fact_checker = ConversableAgent(
    name="fact_checker",
    system_message="You are a fact-checking assistant.",
    llm_config=llm_config,
)

assistant.initiate_chat(
    recipient=fact_checker,
    message="What is AG2?",
    max_turns=2
)
```

### Human-in-the-Loop

Integrate human oversight for critical decisions.  The `human_input_mode` parameter controls how and when human input is requested (ALWAYS, NEVER, TERMINATE).

```python
# Human-in-the-Loop Example
from autogen import ConversableAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig({
    "api_type": "openai",
    "model": "gpt-5-mini",
})

assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

human = UserProxyAgent(
    name="human",
    code_execution_config={"work_dir": "coding", "use_docker": False},
)

human.initiate_chat(
    recipient=assistant,
    message="Hello! What's 2 + 2?"
)
```

### Orchestrating Multiple Agents

Create sophisticated multi-agent systems using flexible orchestration patterns such as `GroupChat` and `Swarm`.

```python
# Multi-Agent Orchestration Example
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

llm_config = LLMConfig({
    "api_type": "openai",
    "model": "gpt-5-mini",
})

# Agent definitions and GroupChat setup...
```

### Tools

Equip agents with tools to access external data, APIs, and functionality.

```python
# Tool Integration Example
from datetime import datetime
from typing import Annotated

from autogen import ConversableAgent, register_function, LLMConfig

llm_config = LLMConfig({
    "api_type": "openai",
    "model": "gpt-5-mini",
})

def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")

date_agent = ConversableAgent(
    name="date_agent",
    system_message="You get the day of the week for a given date.",
    llm_config=llm_config,
)

executor_agent = ConversableAgent(
    name="executor_agent",
    human_input_mode="NEVER",
    llm_config=llm_config,
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

AG2 supports:

*   [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
*   [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
*   [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
*   [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
*   [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)

## Announcements

*   **Nov 11, 2024:** AutoGen evolves into **AG2**!  [AG2AI](https://github.com/ag2ai) hosts the development. Check out [AG2's new look](https://ag2.ai/).
*   **License:**  Apache 2.0 license adopted from v0.3. Enhances open-source collaboration.
*   **May 29, 2024:** DeepLearning.ai launches [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen) course.
*   **May 24, 2024:** Foundation Capital article [The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and video [AI in the Real World Episode 2: Exploring Multi-Agent AI and AutoGen with Chi Wang](https://www.youtube.com/watch?v=RLwyXRVvlNk).
*   **Apr 17, 2024:** Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and [What's next for AI agentic workflows](https://youtu.be/sal78ACtGTc?si=JduUzN_1kDnMq0vF) at Sequoia Capital's AI Ascent (Mar 26).

[More Announcements](announcements.md)

## Contributors Wall

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

## Code Style and Linting

Maintain code quality using pre-commit hooks.

1.  Install: `pip install pre-commit`
2.  Install hooks: `pre-commit install`
3.  Run manually: `pre-commit run --all-files`

## Related Papers

*   [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
*   ...and more (see original README for full list).

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

Licensed under the [Apache License, Version 2.0 (Apache-2.0)](./LICENSE).

This project is a spin-off of [AutoGen](https://github.com/microsoft/autogen).
- The original code from https://github.com/microsoft/autogen is licensed under the MIT License. See the [LICENSE_original_MIT](./license_original/LICENSE_original_MIT) file for details.
- Modifications and additions made in this fork are licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for the full license text.
For more details, please see the [NOTICE](./NOTICE.md) file.