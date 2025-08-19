<p align="center">
  <!-- The image URL points to the GitHub-hosted content, ensuring it displays correctly on the PyPI website.-->
  <img src="https://raw.githubusercontent.com/ag2ai/ag2/27b37494a6f72b1f8050f6bd7be9a7ff232cf749/website/static/img/ag2.svg" width="150" title="hover text">

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


# AG2: The Open-Source AgentOS for Building and Orchestrating AI Agents

AG2 (formerly AutoGen) empowers developers to build and deploy sophisticated AI agents capable of collaboration, tool use, and human interaction; [explore the code on GitHub](https://github.com/ag2ai/ag2).

**Key Features:**

*   **Multi-Agent Collaboration:** Design AI systems where multiple agents interact to solve complex tasks.
*   **LLM and Tool Integration:** Seamlessly integrate various Large Language Models (LLMs) and tools for enhanced functionality.
*   **Human-in-the-Loop Workflows:** Incorporate human input for control and refinement.
*   **Customizable Workflows:** Leverage diverse conversation patterns to create automated and efficient processes.
*   **Open Source & Community Driven:** Fully open-sourced with a welcoming community of collaborators.

## Getting Started

### Installation

Requires Python >= 3.10, < 3.14. Install AG2 using pip:

```bash
pip install ag2[openai]
```

Install extra options based on the features you need.

### Setup Your API Keys

Use the `OAI_CONFIG_LIST` file for secure API key management.  See the sample:

```json
[
  {
    "model": "gpt-4o",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run Your First Agent

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")


with llm_config:
    assistant = AssistantAgent("assistant")
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
# This initiates an automated chat between the two agents to solve the task
```

## Core Concepts

*   **Conversable Agent:** Foundation for communication between AI entities.
*   **Human-in-the-Loop:** Incorporates human feedback for critical tasks.
*   **Orchestrating Multiple Agents:** Facilitates collaboration through built-in and custom patterns.
*   **Tools:** Provides access to external data, APIs, and functionality.
*   **Advanced Agentic Design Patterns:** Includes features like structured outputs, RAG, code execution, etc.

### Conversable Agent

The [ConversableAgent](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent) is the fundamental building block.

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

### Human in the Loop

Integrates human feedback with `human_input_mode`.

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

Enables sophisticated multi-agent collaboration with built-in patterns like `GroupChat`.

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

planner_message = """...Lesson plan details..."""
reviewer_message = """...Reviewer details..."""

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
        system_message="...Teacher details...",
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

Enhance agent capabilities with external tools.

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

Explore additional features in the [documentation](https://docs.ag2.ai/).

## Announcements

*   **Nov 11, 2024:** AutoGen is evolving into AG2! [AG2AI](https://github.com/ag2ai) is the new home.
*   **May 29, 2024:** DeepLearning.ai course "AI Agentic Design Patterns with AutoGen."
*   **May 24, 2024:** Foundation Capital article on Forbes: The Promise of Multi-Agent AI and video.
*   **Apr 17, 2024:** Cited by Andrew Ng in The Batch newsletter and Sequoia Capital's AI Ascent.

[More Announcements](announcements.md)

## Contributors

[![Contributors](https://contrib.rocks/image?repo=ag2ai/ag2&max=204)](https://github.com/ag2ai/ag2/graphs/contributors)

## Code Style and Linting

Follow these steps before contributing:

1.  Install pre-commit: `pip install pre-commit`
2.  Install pre-commit hooks: `pre-commit install`
3.  Run hooks: `pre-commit run --all-files`

## Related Papers

*   AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
*   EcoOptiGen: Hyperparameter Optimization for Large Language Model Generation Inference
*   MathChat: Converse to Tackle Challenging Math Problems with LLM Agents
*   AgentOptimizer: Offline Training of Language Model Agents with Functions as Learnable Weights
*   StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows

## Citing AG2

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

This project is a spin-off of [AutoGen](https://github.com/microsoft/autogen) and contains code under two licenses:
*   Original code from https://github.com/microsoft/autogen is licensed under the MIT License. See the [LICENSE_original_MIT](./license_original/LICENSE_original_MIT) file for details.
*   Modifications and additions made in this fork are licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for the full license text.
For more details, please see the [NOTICE](./NOTICE.md) file.