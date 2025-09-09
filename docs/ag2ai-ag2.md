# AG2: Build Powerful AI Agents with an Open-Source AgentOS

AG2 empowers you to build, experiment with, and deploy AI agents effortlessly, making complex AI tasks achievable with ease. [Explore the AG2 GitHub Repository](https://github.com/ag2ai/ag2).

[<img src="https://raw.githubusercontent.com/ag2ai/ag2/27b37494a6f72b1f8050f6bd7be9a7ff232cf749/website/static/img/ag2.svg" width="100" title="AG2 Logo">](https://github.com/ag2ai/ag2)

[![Downloads per Month](https://static.pepy.tech/personalized-badge/ag2?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/ag2)
[![PyPI Version](https://img.shields.io/pypi/v/ag2?label=PyPI&color=green)](https://pypi.org/project/autogen/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ag2.svg?label=)](https://pypi.org/project/ag2/)
[![Build Status](https://github.com/ag2ai/ag2/actions/workflows/python-package.yml/badge.svg)](https://github.com/ag2ai/ag2/actions/workflows/python-package.yml)
[![Discord](https://img.shields.io/discord/1153072414184452236?logo=discord&style=flat)](https://discord.gg/pAbnFJrkgZ)
[![Follow @ag2ai on X](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ag2ai)](https://x.com/ag2oss)

<p align="center">
  <a href="https://docs.ag2.ai/">📚 Documentation</a> |
  <a href="https://github.com/ag2ai/build-with-ag2">💡 Examples</a> |
  <a href="https://docs.ag2.ai/latest/docs/contributor-guide/contributing">🤝 Contributing</a> |
  <a href="#related-papers">📝 Cite Paper</a> |
  <a href="https://discord.gg/pAbnFJrkgZ">💬 Join Discord</a>
</p>

AG2 is an open-source framework, built from AutoGen, designed to simplify the development and research of AI agents. It enables multi-agent interactions, integration of various LLMs and tools, and supports both autonomous and human-in-the-loop workflows, creating versatile agentic AI solutions.

## Key Features of AG2

*   **Multi-Agent Collaboration:** Facilitates seamless interactions between AI agents, enabling complex task decomposition and problem-solving.
*   **LLM Agnostic:** Supports a wide range of Large Language Models (LLMs), offering flexibility in choosing the best model for your needs.
*   **Tool Integration:** Allows agents to utilize various tools and APIs, expanding their capabilities and real-world applicability.
*   **Human-in-the-Loop Workflows:** Provides mechanisms for human oversight and intervention, enhancing safety and control.
*   **Flexible Orchestration:** Offers built-in conversation patterns (swarms, group chats, etc.) and customization options to tailor agent interactions.
*   **Open Source and Community-Driven:** Developed and maintained by a dynamic group of volunteers, fostering collaboration and innovation.

## Getting Started

For a comprehensive guide to AG2 concepts and implementation, refer to the [Basic Concepts](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/installing-ag2/) section in our documentation.

### Installation

AG2 requires **Python version >= 3.10, < 3.14**. Install AG2 via pip:

```bash
pip install ag2[openai]
```

Install additional dependencies based on your use cases, such as `ag2[azure]` for Azure OpenAI, or similar.

### Setup API Keys

To manage your LLM configurations, we recommend storing API keys in an `OAI_CONFIG_LIST` file.  Here's an example:

```json
[
  {
    "model": "gpt-5",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run Your First Agent

Here's a simple example to get you started:

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

assistant = AssistantAgent("assistant", llm_config=llm_config)

user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})

user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
# This initiates an automated chat between the two agents to solve the task
```

## Example Applications

Explore the following resources to jumpstart your AG2 projects:

*   [Build with AG2](https://github.com/ag2ai/build-with-ag2)
*   [Jupyter Notebooks](notebook)

## Core Agent Concepts

AG2 offers powerful agent concepts to facilitate complex AI development:

*   **Conversable Agent:** Enables message exchange and response generation, forming the foundation for agent communication.
*   **Human-in-the-Loop:** Integrates human input seamlessly for oversight and control.
*   **Orchestrating Multiple Agents:** Provides flexible patterns for multi-agent collaboration, allowing specialized agents to work together.
*   **Tools:** Empowers agents with the ability to utilize external tools and APIs.
*   **Advanced Concepts:** Offers support for structured outputs, Retrieval Augmented Generation (RAG), code execution, and more.

### Conversable Agent

The `ConversableAgent` is the fundamental building block in AG2 for AI communication.

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

### Human in the Loop

Incorporate human oversight into agent workflows with ease. Use `human_input_mode` for customized control:

*   `ALWAYS`: Requires human input for every response.
*   `NEVER`: Operates autonomously.
*   `TERMINATE`: Requests human input only to end conversations.

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

Build collaborative agent systems using flexible orchestration patterns like `GroupChat`.

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

llm_config = LLMConfig({
    "api_type": "openai",
    "model": "gpt-5-mini",
})

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

lesson_planner = ConversableAgent(
    name="planner_agent",
    system_message=planner_message,
    llm_config=llm_config,
)

lesson_reviewer = ConversableAgent(
    name="reviewer_agent",
    system_message=reviewer_message,
    llm_config=llm_config,
)

teacher_message = """You are a classroom teacher.
You decide topics for lessons and work with a lesson planner.
and reviewer to create and finalise lesson plans.
When you are happy with a lesson plan, output "DONE!".
"""

teacher = ConversableAgent(
    name="teacher_agent",
    system_message=teacher_message,
    is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),
    llm_config=llm_config,
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

Equip your agents with tools to extend their capabilities.

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

AG2 also supports:
*   [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
*   [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
*   [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
*   [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
*   [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)

## Announcements

*   **Nov 11, 2024:** AutoGen evolves into **AG2**! The new organization [AG2AI](https://github.com/ag2ai) is created to host the development of AG2 and related projects with open governance. See [AG2's new look](https://ag2.ai/).
*   **License:** AG2 adopts the Apache 2.0 license from v0.3 for enhanced open-source collaboration and user protection.
*   **May 29, 2024:** DeepLearning.ai launched a new short course [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen).
*   **May 24, 2024:** Foundation Capital published an article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and a video [AI in the Real World Episode 2: Exploring Multi-Agent AI and AutoGen with Chi Wang](https://www.youtube.com/watch?v=RLwyXRVvlNk).
*   **Apr 17, 2024:** Andrew Ng cited AutoGen.

[More Announcements](announcements.md)

## Contributors

[<img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" alt="Contributors"/>](https://github.com/ag2ai/ag2/graphs/contributors)

## Code Style and Linting

Use pre-commit hooks to maintain code quality.

1.  Install: `pip install pre-commit`
2.  Install hooks: `pre-commit install`
3.  Run hooks manually: `pre-commit run --all-files`

## Related Papers

*   [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
*   And more, as listed in the original README.

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

This project is a spin-off of [AutoGen](https://github.com/microsoft/autogen) and contains code under two licenses:

*   The original code from https://github.com/microsoft/autogen is licensed under the MIT License. See the [LICENSE_original_MIT](./license_original/LICENSE_original_MIT) file for details.
*   Modifications and additions made in this fork are licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for the full license text.

See the [NOTICE](./NOTICE.md) file for further details.