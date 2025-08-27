<a name="readme-top"></a>

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
  <a href="#related-papers">📝 Cite Paper</a> |
  <a href="https://discord.gg/pAbnFJrkgZ">💬 Join Discord</a>
</p>

<p align="center">
AG2, evolved from AutoGen, is your open-source AgentOS for building and deploying AI agents.
</p>

# AG2: Build Powerful AI Agents with Ease

AG2 is an open-source framework designed to simplify the creation and collaboration of AI agents, offering a robust AgentOS for diverse applications.  [Explore AG2 on GitHub](https://github.com/ag2ai/ag2).

**Key Features:**

*   **Multi-Agent Collaboration:**  Enable seamless interaction between multiple AI agents for complex problem-solving.
*   **LLM Agnostic:**  Integrate with various Large Language Models (LLMs) for flexible development.
*   **Tool Integration:**  Empower agents with tools and external resources for enhanced functionality.
*   **Human-in-the-Loop:**  Incorporate human input and oversight for reliable agent behavior.
*   **Flexible Workflows:**  Design custom or pre-built multi-agent conversation patterns.

AG2 is actively maintained by a dynamic team of volunteers from various organizations.  For maintainer inquiries, contact Chi Wang and Qingyun Wu at [support@ag2.ai](mailto:support@ag2.ai).

## Table of Contents

*   [AG2: Build Powerful AI Agents with Ease](#ag2-build-powerful-ai-agents-with-ease)
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

Dive into AG2 with the [Basic Concepts](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/installing-ag2/) section of our documentation for step-by-step guidance.

### Installation

Ensure you have **Python version >= 3.10, < 3.14**. Install AG2 using pip:

```bash
pip install ag2[openai]
```

Use the `[openai]` option to install the OpenAI dependencies.  For minimal dependencies, install without any extras.

### Setup API Keys

For secure and manageable API key storage, we recommend using the `OAI_CONFIG_LIST` file.

Use `OAI_CONFIG_LIST_sample` as a template:

```json
[
  {
    "model": "gpt-4o",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run Your First Agent

Create a Python script or Jupyter Notebook and run your first agent:

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

with llm_config:
    assistant = AssistantAgent("assistant")
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
```

## Example Applications

Get started with various use cases and explore example applications in our dedicated repository.

*   [Build with AG2](https://github.com/ag2ai/build-with-ag2)
*   [Jupyter Notebooks](notebook)

## Agent Concepts

AG2 offers several agent concepts to streamline the development of your AI agents:

*   **Conversable Agent:** Facilitates message exchange and response generation using GenAI models, tools, or human inputs.
*   **Human-in-the-Loop:** Integrates human input for crucial decisions.
*   **Orchestrating Multiple Agents:** Enables the creation of collaborative systems using pre-built or custom conversation patterns.
*   **Tools:** Equips agents with programs to access external data, APIs, and more.
*   **Advanced Agentic Design Patterns:** Supports structured outputs, RAG, code execution, and more.

### Conversable Agent

The [ConversableAgent](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent) is the core building block for agent communication.

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

*   _Note: Ensure you have your `OPENAI_API_KEY` set as an environment variable before running._

### Human-in-the-Loop

Incorporate human oversight for critical decisions using `human_input_mode`.
AG2 provides the specialized `UserProxyAgent` class.

Example:

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

Create complex multi-agent systems with built-in patterns like `GroupChat`.

Example:

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

planner_message = """...Lesson plan instructions..."""
reviewer_message = """...Review instructions..."""

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

Find more on the `GroupChat` pattern [here](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/group-chat/introduction).

### Tools

Empower agents with tools to access external data and APIs.

Example:

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

Explore further features for enhanced functionality:

*   [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
*   [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
*   [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
*   [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
*   [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)

## Announcements

*   🔥 🎉 **Nov 11, 2024:** AutoGen is evolving into **AG2**! [AG2AI](https://github.com/ag2ai) is created to host the development of AG2. Check [AG2's new look](https://ag2.ai/).
*   📄 **License:** We adopt the Apache 2.0 license from v0.3.
*   🎉 May 29, 2024: DeepLearning.ai launched a new short course [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen).
*   🎉 May 24, 2024: Foundation Capital article [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and video [AI in the Real World Episode 2: Exploring Multi-Agent AI and AutoGen with Chi Wang](https://www.youtube.com/watch?v=RLwyXRVvlNk).
*   🎉 Apr 17, 2024: Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and [What's next for AI agentic workflows](https://youtu.be/sal78ACtGTc?si=JduUzN_1kDnMq0vF).

[More Announcements](announcements.md)

## Contributors Wall

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

## Code Style and Linting

This project enforces code quality using pre-commit hooks.  To contribute:

1.  Install pre-commit:

    ```bash
    pip install pre-commit
    pre-commit install
    ```

2.  The hooks run automatically on commit.  Or, run manually:

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

*   The original code from https://github.com/microsoft/autogen is licensed under the MIT License. See the [LICENSE_original_MIT](./license_original/LICENSE_original_MIT) file for details.
*   Modifications and additions made in this fork are licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for the full license text.

We have documented these changes for clarity and to ensure transparency with our user and contributor community. For more details, please see the [NOTICE](./NOTICE.md) file.