<p align="center">
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

# AG2: Powering the Future of AI Agents

**AG2 is the open-source AgentOS, empowering you to build, orchestrate, and collaborate with AI agents for complex tasks.** Explore the original repository [here](https://github.com/ag2ai/ag2).

## Key Features

*   **Multi-Agent Collaboration:** Facilitate seamless interaction between multiple AI agents, enabling them to work together to solve complex problems.
*   **LLM Agnostic:** Supports a wide range of Large Language Models (LLMs) and easy integration with different providers.
*   **Tool Integration:** Equip agents with the ability to use tools, APIs, and external data sources for enhanced capabilities.
*   **Human-in-the-Loop:** Easily incorporate human oversight and feedback into agent workflows.
*   **Flexible Orchestration:** Offers diverse conversation patterns, including group chats, swarms, and custom orchestration options.

## Table of Contents

*   [Key Features](#key-features)
*   [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Setup API Keys](#setup-your-api-keys)
    *   [Run Your First Agent](#run-your-first-agent)
*   [Example Applications](#example-applications)
*   [Core Agent Concepts](#introduction-of-different-agent-concepts)
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

Begin building your AI agent applications with AG2.

### Installation

Install AG2 using pip:

```bash
pip install ag2[openai]
```

(or `pip install autogen[openai]`)

Install any extra options for additional features.

### Setup API Keys

Configure your LLM dependencies with the `OAI_CONFIG_LIST` file. Use the provided `OAI_CONFIG_LIST_sample` as a template:

```json
[
  {
    "model": "gpt-5",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run Your First Agent

Create a Python script or Jupyter Notebook to run your first agent.

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

assistant = AssistantAgent("assistant", llm_config=llm_config)

user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})

user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
# This initiates an automated chat between the two agents to solve the task
```

## Example Applications

Explore the [Build with AG2](https://github.com/ag2ai/build-with-ag2) repository and example Jupyter notebooks.

## Core Agent Concepts

Learn about essential agent concepts within AG2.

### Conversable Agent

The [ConversableAgent](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent) is the foundation for agent communication.

```python
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

Incorporate human input seamlessly into your agent workflows.

```python
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

Create collaborative systems with flexible orchestration.

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

llm_config = LLMConfig({
    "api_type": "openai",
    "model": "gpt-5-mini",
})

planner_message = """You are a classroom lesson agent. ..."""
reviewer_message = """You are a classroom lesson reviewer. ..."""

lesson_planner = ConversableAgent(name="planner_agent", system_message=planner_message, description="Creates or revises lesson plans.", llm_config=llm_config,)
lesson_reviewer = ConversableAgent(name="reviewer_agent", system_message=reviewer_message, description="Provides one round of reviews to a lesson plan for the lesson_planner to revise.", llm_config=llm_config,)
teacher_message = """You are a classroom teacher. ..."""
teacher = ConversableAgent(name="teacher_agent",system_message=teacher_message,is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),llm_config=llm_config,)

groupchat = GroupChat(agents=[teacher, lesson_planner, lesson_reviewer],speaker_selection_method="auto",messages=[],)
manager = GroupChatManager(name="group_manager",groupchat=groupchat,llm_config=llm_config,)

teacher.initiate_chat(recipient=manager,message="Today, let's introduce our kids to the solar system.")
```

### Tools

Empower agents with tools for enhanced capabilities.

```python
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

date_agent = ConversableAgent(name="date_agent",system_message="You get the day of the week for a given date.",llm_config=llm_config,)
executor_agent = ConversableAgent(name="executor_agent",human_input_mode="NEVER",llm_config=llm_config,)

register_function(get_weekday,caller=date_agent,executor=executor_agent,description="Get the day of the week for a given date",)

chat_result = executor_agent.initiate_chat(recipient=date_agent,message="I was born on the 25th of March 1995, what day was it?",max_turns=2,)
print(chat_result.chat_history[-1]["content"])
```

### Advanced Agentic Design Patterns

*   [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
*   [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
*   [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
*   [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
*   [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)

## Announcements

🔥 🎉 **Nov 11, 2024:** We are evolving AutoGen into **AG2**!
A new organization [AG2AI](https://github.com/ag2ai) is created to host the development of AG2 and related projects with open governance. Check [AG2's new look](https://ag2.ai/).

📄 **License:**
We adopt the Apache 2.0 license from v0.3. This enhances our commitment to open-source collaboration while providing additional protections for contributors and users alike.

🎉 May 29, 2024: DeepLearning.ai launched a new short course [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen), made in collaboration with Microsoft and Penn State University, and taught by AutoGen creators [Chi Wang](https://github.com/sonichi) and [Qingyun Wu](https://github.com/qingyun-wu).

🎉 May 24, 2024: Foundation Capital published an article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and a video [AI in the Real World Episode 2: Exploring Multi-Agent AI and AutoGen with Chi Wang](https://www.youtube.com/watch?v=RLwyXRVvlNk).

🎉 Apr 17, 2024: Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and [What's next for AI agentic workflows](https://youtu.be/sal78ACtGTc?si=JduUzN_1kDnMq0vF) at Sequoia Capital's AI Ascent (Mar 26).

[More Announcements](announcements.md)

## Contributors Wall

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

## Code Style and Linting

This project uses pre-commit hooks.

1.  Install:

```bash
pip install pre-commit
pre-commit install
```

2.  Run:

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

Licensed under the [Apache License, Version 2.0 (Apache-2.0)](./LICENSE).

*   Original code from https://github.com/microsoft/autogen is licensed under the MIT License. See the [LICENSE\_original\_MIT](./license_original/LICENSE_original_MIT) file.
*   Modifications and additions are licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file.

See [NOTICE](./NOTICE.md) for more details.