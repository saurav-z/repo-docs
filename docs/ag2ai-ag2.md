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

## AG2: Powering the Future of AI Agentic Workflows

AG2 is an open-source framework, evolved from AutoGen, that empowers developers to build, experiment with, and deploy AI agents and multi-agent systems with ease.  [Explore the AG2 Repository on GitHub!](https://github.com/ag2ai/ag2)

**Key Features:**

*   **Multi-Agent Collaboration:**  Orchestrate AI agents to work together seamlessly, using built-in conversation patterns like group chats and swarms, or customize your own.
*   **Conversable Agents:**  Build agents that can send, receive, and respond to messages, leveraging LLMs, non-LLM tools, and human input.
*   **Human-in-the-Loop:**  Incorporate human oversight and feedback into your agent workflows for critical decision-making and improved results.
*   **Tool Integration:**  Equip agents with powerful tools, including external APIs and custom functions, to expand their capabilities.
*   **Advanced Agentic Design:** Supports structured outputs, RAG, code execution, and more.

## Getting Started

### Installation

Ensure you have Python 3.10 to 3.13 installed. Then install AG2:

```bash
pip install ag2[openai]
```
Install extra dependencies as needed based on desired features.

### Setup your API keys

Use the `OAI_CONFIG_LIST_sample` file as a template, and store your API keys in a JSON file:

```json
[
  {
    "model": "gpt-4o",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run your first agent

Get started quickly with a simple agent example.

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

Dive into practical use cases and explore AG2's capabilities through our examples and Jupyter notebooks:

*   [Build with AG2](https://github.com/ag2ai/build-with-ag2)
*   [Jupyter Notebooks](notebook)

## Core Agent Concepts

AG2 offers key agent types to build your AI applications:

*   **Conversable Agent:** The foundation for AI agents that can communicate and generate responses.
*   **Human-in-the-Loop:** Integrate human input for control and oversight.
*   **Orchestrating Multiple Agents:** Create complex workflows with coordinated agent interactions.
*   **Tools:** Empower agents with external resources and functionality.
*   **Advanced Concepts:** Enhance your agents with structured outputs, RAG, and more.

### Conversable Agent

The `ConversableAgent` is the core of AG2, enabling easy communication between AI entities.

Example:

```python
# 1. Import ConversableAgent class
from autogen import ConversableAgent, LLMConfig

# 2. Define our LLM configuration for OpenAI's GPT-4o mini
#    uses the OPENAI_API_KEY environment variable
llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")


# 3. Create our LLM agent
with llm_config:
  # Create an AI agent
  assistant = ConversableAgent(
      name="assistant",
      system_message="You are an assistant that responds concisely.",
  )

  # Create another AI agent
  fact_checker = ConversableAgent(
      name="fact_checker",
      system_message="You are a fact-checking assistant.",
  )

# 4. Start the conversation
assistant.initiate_chat(
    recipient=fact_checker,
    message="What is AG2?",
    max_turns=2
)
```

### Human-in-the-Loop

Incorporate human feedback for control and improved results.

Example:

```python
# 1. Import ConversableAgent and UserProxyAgent classes
from autogen import ConversableAgent, UserProxyAgent, LLMConfig

# 2. Define our LLM configuration for OpenAI's GPT-4o mini
#    uses the OPENAI_API_KEY environment variable
llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")


# 3. Create our LLM agent
with llm_config:
  assistant = ConversableAgent(
      name="assistant",
      system_message="You are a helpful assistant.",
  )

# 4. Create a human agent with manual input mode
human = ConversableAgent(
    name="human",
    human_input_mode="ALWAYS"
)
# or
human = UserProxyAgent(name="human", code_execution_config={"work_dir": "coding", "use_docker": False})

# 5. Start the chat
human.initiate_chat(
    recipient=assistant,
    message="Hello! What's 2 + 2?"
)
```

### Orchestrating Multiple Agents

Build collaborative systems where specialized agents work together.

Example:

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

# Put your key in the OPENAI_API_KEY environment variable
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

# 1. Add a separate 'description' for our planner and reviewer agents
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

# 2. The teacher's system message can also be used as a description, so we don't define it
teacher_message = """You are a classroom teacher.
You decide topics for lessons and work with a lesson planner.
and reviewer to create and finalise lesson plans.
When you are happy with a lesson plan, output "DONE!".
"""

with llm_config:
    teacher = ConversableAgent(
        name="teacher_agent",
        system_message=teacher_message,
        # 3. Our teacher can end the conversation by saying DONE!
        is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),
    )

# 4. Create the GroupChat with agents and selection method
groupchat = GroupChat(
    agents=[teacher, lesson_planner, lesson_reviewer],
    speaker_selection_method="auto",
    messages=[],
)

# 5. Our GroupChatManager will manage the conversation and uses an LLM to select the next agent
manager = GroupChatManager(
    name="group_manager",
    groupchat=groupchat,
    llm_config=llm_config,
)

# 6. Initiate the chat with the GroupChatManager as the recipient
teacher.initiate_chat(
    recipient=manager,
    message="Today, let's introduce our kids to the solar system."
)
```

### Tools

Extend agents' capabilities with external tools.

Example:

```python
from datetime import datetime
from typing import Annotated

from autogen import ConversableAgent, register_function, LLMConfig

# Put your key in the OPENAI_API_KEY environment variable
llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

# 1. Our tool, returns the day of the week for a given date
def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")

# 2. Agent for determining whether to run the tool
with llm_config:
    date_agent = ConversableAgent(
        name="date_agent",
        system_message="You get the day of the week for a given date.",
    )

# 3. And an agent for executing the tool
executor_agent = ConversableAgent(
    name="executor_agent",
    human_input_mode="NEVER",
)

# 4. Registers the tool with the agents, the description will be used by the LLM
register_function(
    get_weekday,
    caller=date_agent,
    executor=executor_agent,
    description="Get the day of the week for a given date",
)

# 5. Two-way chat ensures the executor agent follows the suggesting agent
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

*   **Nov 11, 2024:**  AutoGen evolves into AG2!  [AG2AI](https://github.com/ag2ai) is the new home.  Check out [AG2's new look](https://ag2.ai/).
*   **License:** Apache 2.0 from v0.3.  More open and contributor-friendly.
*   **May 29, 2024:** DeepLearning.ai course on AI Agentic Design Patterns with AutoGen.
*   **May 24, 2024:**  Foundation Capital article on multi-agent AI (Forbes) and video.
*   **Apr 17, 2024:**  Andrew Ng cites AutoGen in The Batch newsletter.

[More Announcements](announcements.md)

## Contributors

See the vibrant community contributing to AG2:

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

## Code Style and Linting

This project uses pre-commit hooks for code quality.  To contribute:

1.  Install pre-commit:

    ```bash
    pip install pre-commit
    pre-commit install
    ```

2.  Hooks run automatically on commit, or run manually:

    ```bash
    pre-commit run --all-files
    ```

## Related Papers

*   [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
*   ... (and other papers listed in original README)

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

AG2 is licensed under the [Apache License, Version 2.0 (Apache-2.0)](./LICENSE).

This project builds upon [AutoGen](https://github.com/microsoft/autogen) and includes code under both the MIT License (from AutoGen) and the Apache License 2.0 (for modifications and additions).  See [LICENSE_original_MIT](./license_original/LICENSE_original_MIT) and [LICENSE](./LICENSE) for details.  See [NOTICE](./NOTICE.md) for more information.