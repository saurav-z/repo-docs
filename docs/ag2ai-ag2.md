<p align="center">
  <img src="https://raw.githubusercontent.com/ag2ai/ag2/27b37494a6f72b1f8050f6bd7be9a7ff232cf749/website/static/img/ag2.svg" width="150" title="AG2 Logo">
</p>

<div align="center">
  <a href="https://www.pepy.tech/projects/ag2">
    <img src="https://static.pepy.tech/personalized-badge/ag2?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month" alt="Downloads"/>
  </a>
  <a href="https://pypi.org/project/ag2">
    <img src="https://img.shields.io/pypi/v/ag2?label=PyPI&color=green">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/ag2.svg?label=">
  <a href="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml">
    <img src="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml/badge.svg">
  </a>
  <a href="https://discord.gg/pAbnFJrkgZ">
    <img src="https://img.shields.io/discord/1153072414184452236?logo=discord&style=flat">
  </a>
  <a href="https://x.com/ag2oss">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ag2ai">
  </a>
</div>
<div align="center">
  <a href="https://docs.ag2.ai/">📚 Documentation</a> |
  <a href="https://github.com/ag2ai/build-with-ag2">💡 Examples</a> |
  <a href="https://docs.ag2.ai/latest/docs/contributor-guide/contributing">🤝 Contributing</a> |
  <a href="#related-papers">📝 Cite paper</a> |
  <a href="https://discord.gg/pAbnFJrkgZ">💬 Join Discord</a>
</div>

<h1 align="center">AG2: Build Powerful AI Agents with Ease</h1>

AG2, evolved from AutoGen, is an open-source framework that empowers developers to build, experiment with, and deploy AI agents quickly and efficiently.

<p align="center">
  <a href="https://github.com/ag2ai/ag2">
    <img src="https://img.shields.io/badge/View%20on%20GitHub-1a73e8?style=for-the-badge&logo=github" alt="View on GitHub"/>
  </a>
</p>

## Key Features

*   **Multi-Agent Collaboration:** Design and orchestrate interactions between multiple AI agents to solve complex tasks.
*   **LLM Agnostic:** Seamlessly integrate with various Large Language Models (LLMs).
*   **Tool Integration:** Equip agents with tools to access external data, APIs, and functions.
*   **Human-in-the-Loop Workflows:** Incorporate human input for validation and guidance.
*   **Flexible Conversation Patterns:** Utilize pre-built patterns or create custom orchestration for agents.

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
*   [Code Style and Linting](#code-style-and-linting)
*   [Related Papers](#related-papers)
*   [Contributors Wall](#contributors-wall)
*   [Cite the Project](#cite-the-project)
*   [License](#license)

## Getting Started

Dive into AG2 and build your first AI agent! For detailed guidance, explore the [Basic Concepts](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/installing-ag2/) documentation.

### Installation

AG2 requires Python 3.10 to 3.14. Install AG2 using pip:

```bash
pip install ag2[openai]
```

or for Mac

```bash
pip install 'ag2[openai]'
```

### Setup API Keys

To keep your LLM setup neat and avoid accidental exposure of your API key, store your keys in a configuration file.

Example `OAI_CONFIG_LIST`:

```json
[
  {
    "model": "gpt-5",
    "api_key": "<your OpenAI API key here>"
  }
]
```

Make sure to add your config file to `.gitignore`.

### Run Your First Agent

Create a Python script or Jupyter Notebook to get started:

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})

user_proxy.run(assistant, message="Summarize the main differences between Python lists and tuples.").process()
```

## Example Applications

Explore practical applications and get inspired with our extensive range of examples:

*   [Build with AG2](https://github.com/ag2ai/build-with-ag2)
*   [Jupyter Notebooks](notebook)

## Core Agent Concepts

AG2 provides core agent concepts to help build effective AI agents:

### Conversable Agent

The [ConversableAgent](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent) is the foundation for all AG2 agents, enabling message exchange and response generation.

Here's a basic example of two agents: a coder and a reviewer.

```python
import logging
from autogen import ConversableAgent, LLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

coder = ConversableAgent(
    name="coder",
    system_message="You are a Python developer. Write short Python scripts.",
    llm_config=llm_config,
)

reviewer = ConversableAgent(
    name="reviewer",
    system_message="You are a code reviewer. Analyze provided code and suggest improvements. "
                   "Do not generate code, only suggest improvements.",
    llm_config=llm_config,
)

response = reviewer.run(
            recipient=coder,
            message="Write a Python function that computes Fibonacci numbers.",
            max_turns=10
        )
response.process()
logger.info("Final output:\n%s", response.summary)
```

### Human-in-the-Loop

Use the `UserProxyAgent` to incorporate human feedback:

```python
import logging
from autogen import ConversableAgent, LLMConfig, UserProxyAgent
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

planner_message = "You are a classroom lesson planner. Given a topic, write a lesson plan for a fourth grade class."
reviewer_message = "You are a classroom lesson reviewer. Compare the plan to the curriculum and suggest up to 3 improvements."
teacher_message = "You are an experienced classroom teacher. You don't prepare plans, you provide simple guidance to the planner to prepare a lesson plan on the key topic."

lesson_planner = ConversableAgent(
    name="planner_agent",
    system_message=planner_message,
    description="Creates or revises lesson plans before having them reviewed.",
    is_termination_msg=lambda x: "APPROVED" in (x.get("content", "") or "").upper(),
    human_input_mode="NEVER",
    llm_config=llm_config,
)

lesson_reviewer = ConversableAgent(
    name="reviewer_agent",
    system_message=reviewer_message,
    description="Provides one round of feedback to lesson plans back to the lesson planner before requiring the human validator.",
    llm_config=llm_config,
)

teacher = ConversableAgent(
    name="teacher_agent",
    system_message=teacher_message,
    description="Provides guidance on the topic and content, if required.",
    llm_config=llm_config,
)

human_validator = UserProxyAgent(
    name="human_validator",
    system_message="You are a human educator who provides final approval for lesson plans.",
    description="Evaluates the proposed lesson plan and either approves it or requests revisions, before returning to the planner.",
)

auto_selection = AutoPattern(
    agents=[teacher, lesson_planner, lesson_reviewer],
    initial_agent=teacher,
    user_agent=human_validator,
    group_manager_args={"name": "group_manager", "llm_config": llm_config},
)

response = run_group_chat(
    pattern=auto_selection,
    messages="Let's introduce our kids to the solar system.",
    max_rounds=20,
)

response.process()
logger.info("Final output:\n%s", response.summary)
```

### Orchestrating Multiple Agents

AG2 enables you to create sophisticated multi-agent systems. Example with lesson planner, reviewer and teacher agents.

```python
import logging
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

planner_message = "You are a classroom lesson planner. Given a topic, write a lesson plan for a fourth grade class."
reviewer_message = "You are a classroom lesson reviewer. Compare the plan to the curriculum and suggest up to 3 improvements."
lesson_planner = ConversableAgent(
    name="planner_agent",
    system_message=planner_message,
    description="Creates or revises lesson plans.",
    llm_config=llm_config,
)

lesson_reviewer = ConversableAgent(
    name="reviewer_agent",
    system_message=reviewer_message,
    description="Provides one round of feedback to lesson plans.",
    llm_config=llm_config,
)

teacher_message = "You are a classroom teacher. You decide topics and collaborate with planner and reviewer to finalize lesson plans. When satisfied, output DONE!"
teacher = ConversableAgent(
    name="teacher_agent",
    system_message=teacher_message,
    is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),
    llm_config=llm_config,
)

auto_selection = AutoPattern(
    agents=[teacher, lesson_planner, lesson_reviewer],
    initial_agent=lesson_planner,
    group_manager_args={"name": "group_manager", "llm_config": llm_config},
)

response = run_group_chat(
    pattern=auto_selection,
    messages="Let's introduce our kids to the solar system.",
    max_rounds=20,
)

response.process()
logger.info("Final output:\n%s", response.summary)
```

### Tools

Empower agents with tools for enhanced functionality.

```python
import logging
from datetime import datetime
from typing import Annotated
from autogen import ConversableAgent, register_function, LLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Tool: returns weekday for a given date
def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")

date_agent = ConversableAgent(
    name="date_agent",
    system_message="You find the day of the week for a given date.",
    llm_config=llm_config,
)

executor_agent = ConversableAgent(
    name="executor_agent",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

# Register tool
register_function(
    get_weekday,
    caller=date_agent,
    executor=executor_agent,
    description="Get the day of the week for a given date",
)

# Use tool in chat
chat_result = executor_agent.initiate_chat(
    recipient=date_agent,
    message="I was born on 1995-03-25, what day was it?",
    max_turns=2,
)

logger.info("Final output:\n%s", chat_result.chat_history[-1]["content"])
```

### Advanced Agentic Design Patterns

AG2 also supports more advanced concepts to help you build your AI agent workflows. You can find more information in the documentation.

*   [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
*   [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
*   [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
*   [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
*   [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)
*   [Pattern Cookbook (9 group orchestrations)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/pattern-cookbook/overview/)

## Announcements

🔥 🎉 **Nov 11, 2024:** We are evolving AutoGen into **AG2**! A new organization [AG2AI](https://github.com/ag2ai) is created to host the development of AG2 and related projects with open governance. Check [AG2's new look](https://ag2.ai/).

📄 **License:** We adopt the Apache 2.0 license from v0.3. This enhances our commitment to open-source collaboration while providing additional protections for contributors and users alike.

🎉 May 29, 2024: DeepLearning.ai launched a new short course [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen), made in collaboration with Microsoft and Penn State University, and taught by AutoGen creators [Chi Wang](https://github.com/sonichi) and [Qingyun Wu](https://github.com/qingyun-wu).

🎉 May 24, 2024: Foundation Capital published an article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and a video [AI in the Real World Episode 2: Exploring Multi-Agent AI and AutoGen with Chi Wang](https://www.youtube.com/watch?v=RLwyXRVvlNk).

🎉 Apr 17, 2024: Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and [What's next for AI agentic workflows](https://youtu.be/sal78ACtGTc?si=JduUzN_1kDnMq0vF) at Sequoia Capital's AI Ascent (Mar 26).

[More Announcements](announcements.md)

## Code Style and Linting

Ensure code quality by using pre-commit hooks.

1.  Install:

    ```bash
    pip install pre-commit
    pre-commit install
    ```

2.  Run hooks:

    ```bash
    pre-commit run --all-files
    ```

## Related Papers

*   [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
*   [EcoOptiGen: Hyperparameter Optimization for Large Language Model Generation Inference](https://arxiv.org/abs/2303.04673)
*   [MathChat: Converse to Tackle Challenging Math Problems with LLM Agents](https://arxiv.org/abs/2306.01337)
*   [AgentOptimizer: Offline Training of Language Model Agents with Functions as Learnable Weights](https://arxiv.org/pdf/2402.11359)
*   [StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows](https://arxiv.org/abs/2403.11322)

## Contributors Wall

[Contributors Wall](https://github.com/ag2ai/ag2/graphs/contributors)

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

Licensed under the [Apache License, Version 2.0](./LICENSE).

This project is a spin-off of [AutoGen](https://github.com/microsoft/autogen) and contains code under two licenses:

*   The original code from https://github.com/microsoft/autogen is licensed under the MIT License. See the [LICENSE_original_MIT](./license_original/LICENSE_original_MIT) file for details.
*   Modifications and additions made in this fork are licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for the full license text.

For more details, please see the [NOTICE](./NOTICE.md) file.