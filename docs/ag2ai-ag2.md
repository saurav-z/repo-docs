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

# AG2: Revolutionizing AI Agent Development with Open-Source Innovation

AG2 (formerly AutoGen) is an open-source framework that empowers developers to build and orchestrate AI agents for complex tasks, with the goal of streamlining agentic AI development and research.  Check out the original repo: [https://github.com/ag2ai/ag2](https://github.com/ag2ai/ag2).

**Key Features:**

*   **Multi-Agent Collaboration:** Facilitates seamless interaction and cooperation among multiple AI agents.
*   **LLM & Tool Support:**  Integrates with various Large Language Models (LLMs) and supports tool usage.
*   **Autonomous & Human-in-the-Loop Workflows:** Enables both automated and human-assisted agentic workflows.
*   **Flexible Conversation Patterns:** Offers a range of multi-agent conversation patterns, including swarms, group chats, and more.
*   **Extensible:** Provides a flexible and extensible platform to build new AI agent functionalities and patterns.

## Table of Contents

*   [AG2: Revolutionizing AI Agent Development with Open-Source Innovation](#ag2-revolutionizing-ai-agent-development-with-open-source-innovation)
    *   [Key Features:](#key-features)
    *   [Table of Contents](#table-of-contents)
    *   [Getting Started](#getting-started)
        *   [Installation](#installation)
        *   [Setup Your API Keys](#setup-your-api-keys)
        *   [Run Your First Agent](#run-your-first-agent)
    *   [Example Applications](#example-applications)
    *   [Core Agent Concepts](#core-agent-concepts)
        *   [Conversable Agent](#conversable-agent)
        *   [Orchestrating Multiple Agents](#orchestrating-multiple-agents)
        *   [Human in the Loop](#human-in-the-loop)
        *   [Tools](#tools)
        *   [Advanced Agentic Design Patterns](#advanced-agentic-design-patterns)
    *   [Announcements](#announcements)
    *   [Code Style and Linting](#code-style-and-linting)
    *   [Related Papers](#related-papers)
    *   [Contributors Wall](#contributors-wall)
    *   [Cite the Project](#cite-the-project)
    *   [License](#license)

## Getting Started

Get up and running with AG2 quickly with our easy-to-follow setup guide.  For a step-by-step walk through of AG2 concepts and code, see [Basic Concepts](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/installing-ag2/) in our documentation.

### Installation

AG2 requires **Python version >= 3.10, < 3.14**.  Install AG2 using `ag2` (or its alias `autogen`) on PyPI.

**Windows/Linux:**

```bash
pip install ag2[openai]
```

**Mac:**

```bash
pip install 'ag2[openai]'
```

Minimal dependencies are installed by default.  Install extra options based on your needs.

### Setup Your API Keys

Securely manage your LLM API keys by storing them in a configuration file. This approach prevents accidental key exposure in your codebase.

Example `OAI_CONFIG_LIST` file:

```json
[
  {
    "model": "gpt-5",
    "api_key": "<your OpenAI API key here>"
  }
]
```

**Important:** Add your config file to your `.gitignore` to avoid accidental commits of sensitive information.

### Run Your First Agent

Create a script or Jupyter Notebook to execute your first AG2 agent:

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

assistant = AssistantAgent("assistant", llm_config=llm_config)

user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})

user_proxy.run(assistant, message="Summarize the main differences between Python lists and tuples.").process()
```

## Example Applications

Explore practical implementations of AG2 with our dedicated example applications repository:

*   [Build with AG2](https://github.com/ag2ai/build-with-ag2)
*   [Jupyter Notebooks](notebook)

## Core Agent Concepts

AG2 offers several agent concepts to help you build your AI agents. Here are some of the most common:

### Conversable Agent

The `ConversableAgent` is the foundational component of AG2, enabling AI entities to communicate and generate responses. This core agent type facilitates message exchange, serving as the base class for all agents in the framework.

**Example: Coder and Reviewer Agents**

*   **Coder Agent:** Writes Python code.
*   **Reviewer Agent:** Critiques the code without rewriting it.

```python
import logging
from autogen import ConversableAgent, LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load LLM configuration
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Define agents
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

# Start a conversation
response = reviewer.run(
            recipient=coder,
            message="Write a Python function that computes Fibonacci numbers.",
            max_turns=10
        )

response.process()

logger.info("Final output:\n%s", response.summary)
```

---

### Orchestrating Multiple Agents

AG2 allows you to orchestrate multi-agent systems where specialized agents collaborate to solve complex problems through flexible orchestration patterns.

**Example: Teacher, Lesson Planner, and Reviewer**

*   **Teacher:** Guides the lesson planning process.
*   **Lesson Planner:** Creates the lesson plan.
*   **Lesson Reviewer:** Provides feedback on the lesson plan.

```python
import logging
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Define lesson planner and reviewer
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

---

### Human in the Loop

Incorporate human oversight into your AI workflows for validation and guidance with the `UserProxyAgent`.

**Example: Human Validator**

Extending the **teacher-planner-reviewer** example to include a **human agent** who validates the final lesson plan:

```python
import logging
from autogen import ConversableAgent, LLMConfig, UserProxyAgent
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Same agents as before, but now the human validator will pass to the planner who will check for "APPROVED" and terminate
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

---

### Tools

Enhance agent capabilities with **tools** that extend them with external data, APIs, or functions.

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

---

### Advanced Agentic Design Patterns

AG2 supports advanced concepts to enhance your AI agent workflows.

*   [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
*   [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
*   [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
*   [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
*   [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)
*   [Pattern Cookbook (9 group orchestrations)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/pattern-cookbook/overview/)

## Announcements

*   🔥 🎉 **Nov 11, 2024:** AutoGen is evolving into **AG2**!  A new organization [AG2AI](https://github.com/ag2ai) has been created.
*   📄 **License:**  AG2 adopts the Apache 2.0 license from v0.3.
*   🎉 May 29, 2024: DeepLearning.ai launched a new short course [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen).
*   🎉 May 24, 2024: Foundation Capital published an article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97).
*   🎉 Apr 17, 2024: Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/).

[More Announcements](announcements.md)

## Code Style and Linting

This project enforces code quality with pre-commit hooks.

1.  Install pre-commit:

```bash
pip install pre-commit
pre-commit install
```

2.  Run hooks automatically on commit or manually:

```bash
pre-commit run --all-files
```

## Related Papers

*   [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
*   ... (and other related papers)

## Contributors Wall

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

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