<a name="readme-top"></a>

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

# AG2: Build Powerful AI Agents with an Open-Source AgentOS

**AG2 is an open-source framework that empowers developers to create, orchestrate, and deploy multi-agent AI systems.** Evolved from AutoGen, AG2 streamlines the development of agentic AI, offering a flexible platform for building intelligent applications.  Find the original repository [here](https://github.com/ag2ai/ag2).

## Key Features

*   **Multi-Agent Collaboration:** Design agents that communicate and cooperate to solve complex tasks.
*   **LLM & Tool Integration:** Seamlessly utilize various large language models (LLMs) and tools for diverse functionalities.
*   **Flexible Workflows:** Implement autonomous and human-in-the-loop workflows to optimize agent performance.
*   **Conversation Patterns:** Leverage pre-built and customizable multi-agent conversation patterns for efficient interaction.
*   **Open Source & Collaborative:** Built with open governance, welcoming contributions from all organizations.

## Getting Started

Get up and running with AG2 quickly.

### Installation

AG2 requires **Python version >= 3.10, < 3.14**. Install it using pip:

```bash
pip install ag2[openai]
```

Install optional dependencies based on the specific features you need.

### Setup API Keys

For streamlined LLM management, use the `OAI_CONFIG_LIST` file to store your API keys.

```json
[
  {
    "model": "gpt-4o",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run Your First Agent

Create a Python script or Jupyter Notebook to get started:

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

Explore practical applications and use cases:

*   [Build with AG2](https://github.com/ag2ai/build-with-ag2)
*   [Jupyter Notebooks](notebook)

## Agent Concepts

AG2 provides core agent types and concepts to build AI agents.

*   **Conversable Agent:** The foundation for message exchange and response generation. See the [ConversableAgent documentation](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent)
*   **Human-in-the-Loop:** Incorporate human input for critical decisions.
*   **Orchestration:** Manage multi-agent collaboration with built-in patterns.
*   **Tools:** Empower agents with external data, APIs, and functionality.

### Conversable Agent

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
# or
human = UserProxyAgent(name="human", code_execution_config={"work_dir": "coding", "use_docker": False})

human.initiate_chat(
    recipient=assistant,
    message="Hello! What's 2 + 2?"
)
```

### Orchestrating Multiple Agents

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

planner_description = "Creates or revises lesson plans."
reviewer_description = """Provides one round of reviews to a lesson plan
for the lesson_planner to revise."""

with llm_config:
    lesson_planner = ConversableAgent(
        name="planner_agent",
        system_message=planner_message,
        description=planner_description,
    )

    lesson_reviewer = ConversableAgent(
        name="reviewer_agent",
        system_message=reviewer_message,
        description=reviewer_description,
    )

with llm_config:
    teacher = ConversableAgent(
        name="teacher_agent",
        system_message=teacher_message,
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

*   [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
*   [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
*   [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
*   [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
*   [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)

## Announcements

*   **Nov 11, 2024:** AutoGen evolves into **AG2**!  A new organization [AG2AI](https://github.com/ag2ai) now hosts AG2 development. Learn more at [AG2's new look](https://ag2.ai/).
*   **License:** AG2 adopts the Apache 2.0 license from v0.3.
*   **May 29, 2024:** DeepLearning.ai launched a new short course [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen).
*   **May 24, 2024:** Foundation Capital published an article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and a video [AI in the Real World Episode 2: Exploring Multi-Agent AI and AutoGen with Chi Wang](https://www.youtube.com/watch?v=RLwyXRVvlNk).
*   **Apr 17, 2024:** Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and [What's next for AI agentic workflows](https://youtu.be/sal78ACtGTc?si=JduUzN_1kDnMq0vF) at Sequoia Capital's AI Ascent (Mar 26).

[More Announcements](announcements.md)

## Contributors Wall

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

## Code Style and Linting

This project uses pre-commit hooks for code quality.
1.  Install: `pip install pre-commit`
2.  Run:
    *   On commit: The hooks run automatically.
    *   Manually: `pre-commit run --all-files`

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