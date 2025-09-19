<p align="center">
  <a href="https://github.com/ag2ai/ag2">
    <img src="https://raw.githubusercontent.com/ag2ai/ag2/27b37494a6f72b1f8050f6bd7be9a7ff232cf749/website/static/img/ag2.svg" width="150" alt="AG2 Logo">
  </a>
</p>

<p align="center">
  <a href="https://www.pepy.tech/projects/ag2">
    <img src="https://static.pepy.tech/personalized-badge/ag2?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month" alt="Downloads/Month">
  </a>
  <a href="https://pypi.org/project/autogen/">
    <img src="https://img.shields.io/pypi/v/ag2?label=PyPI&color=green" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/ag2.svg?label=" alt="Python Versions">
  <a href="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml">
    <img src="https://github.com/ag2ai/ag2/actions/workflows/python-package.yml/badge.svg" alt="Build Status">
  </a>
  <a href="https://discord.gg/pAbnFJrkgZ">
    <img src="https://img.shields.io/discord/1153072414184452236?logo=discord&style=flat" alt="Discord">
  </a>
  <a href="https://x.com/ag2oss">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ag2ai" alt="Follow @ag2ai on Twitter">
  </a>
</p>

<p align="center">
  <a href="https://docs.ag2.ai/">📚 Documentation</a> |
  <a href="https://github.com/ag2ai/build-with-ag2">💡 Examples</a> |
  <a href="https://docs.ag2.ai/latest/docs/contributor-guide/contributing">🤝 Contributing</a> |
  <a href="#related-papers">📝 Cite Paper</a> |
  <a href="https://discord.gg/pAbnFJrkgZ">💬 Join Discord</a>
</p>

# AG2: Build and Orchestrate AI Agents with Ease

AG2 (formerly AutoGen) is an open-source framework empowering you to build, customize, and deploy AI agents, enabling complex task solving through multi-agent collaboration and offering streamlined workflows.

## Key Features

*   **Multi-Agent Collaboration:** Facilitates communication and cooperation among AI agents to tackle complex tasks.
*   **LLM Agnostic:** Supports various Large Language Models (LLMs), offering flexibility in model selection.
*   **Tool Integration:** Enables agents to utilize tools and APIs, expanding their capabilities.
*   **Human-in-the-Loop Workflows:** Incorporates human input for validation and guidance.
*   **Flexible Orchestration:** Offers diverse conversation patterns (swarms, group chats, etc.) and customization options.

## Table of Contents

*   [Key Features](#key-features)
*   [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Setting up API Keys](#setup-your-api-keys)
    *   [Run Your First Agent](#run-your-first-agent)
*   [Example Applications](#example-applications)
*   [Agent Concepts Explained](#introduction-of-different-agent-concepts)
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

To begin using AG2 and explore its concepts, refer to the [Basic Concepts](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/installing-ag2/) section of our documentation for a step-by-step guide.

### Installation

AG2 requires **Python version >= 3.10, < 3.14**. Install AG2 via `ag2` (or its alias `autogen`) from PyPI.

**Windows/Linux:**

```bash
pip install ag2[openai]
```

**Mac:**

```bash
pip install 'ag2[openai]'
```

By default, only minimal dependencies are installed. You can include optional dependencies based on the features you need.

### Setting up API Keys

To securely manage your LLM API keys and avoid accidental commits, we recommend storing your keys in a configuration file.

In our examples, we use a file named **`OAI_CONFIG_LIST`** to store API keys. While you can choose any filename, remember to add it to `.gitignore` to prevent it from being tracked by source control.

Use the following template for your configuration file:

```json
[
  {
    "model": "gpt-5",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run Your First Agent

Create a script or Jupyter Notebook and execute your first agent.

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

assistant = AssistantAgent("assistant", llm_config=llm_config)

user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})

user_proxy.run(assistant, message="Summarize the main differences between Python lists and tuples.").process()
```

## Example Applications

Explore a wide range of practical applications to get started with AG2. The [Build with AG2](https://github.com/ag2ai/build-with-ag2) repository offers numerous examples, or you can begin with our collection of Jupyter notebooks.

-   [Build with AG2](https://github.com/ag2ai/build-with-ag2)
-   [Jupyter Notebooks](notebook)

## Agent Concepts Explained

AG2 offers several core agent concepts to help you build your AI agent workflows.

*   **Conversable Agent:** The fundamental building block for agent communication, capable of sending, receiving, and generating responses using GenAI models, tools, or human input.
*   **Human-in-the-Loop:** Allows human input and validation during the conversation
*   **Orchestrating Multiple Agents:** Enables the orchestration of multiple agents using built-in conversation patterns (swarms, group chats, etc.) or custom reply methods.
*   **Tools:** Provides tools that can be registered, invoked, and executed by agents.
*   **Advanced Concepts:** Includes structured outputs, Retrieval Augmented Generation (RAG), code execution, and more.

### Conversable Agent

The [ConversableAgent](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent) is the base class designed for seamless AI interactions. It handles message exchanges and response generation.

Here's a simple example with two agents:

*   A **coder agent** to write Python code.
*   A **reviewer agent** to critique the code.

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

AG2 supports the creation of multi-agent systems where specialized agents collaborate to solve complex problems through flexible orchestration.

Here's an example with a **teacher**, **lesson planner**, and **reviewer** agents working together:

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

### Human-in-the-Loop

Incorporate human oversight to validate or guide AI outputs using the `UserProxyAgent`.

Extend the previous example to include a **human agent** to validate the final lesson:

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

Agents gain significant utility through **tools**, which extend their capabilities with external data, APIs, or functions.

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

AG2 supports more advanced concepts to help you build your AI agent workflows.

-   [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
-   [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
-   [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
-   [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
-   [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)
-   [Pattern Cookbook (9 group orchestrations)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/pattern-cookbook/overview/)

## Announcements

🔥 🎉 **November 11, 2024:** AutoGen evolves into **AG2**! Check out [AG2's new look](https://ag2.ai/).

📄 **License:** We adopted the Apache 2.0 license from v0.3 to enhance our commitment to open-source collaboration and protect contributors and users.

🎉 **May 29, 2024:** DeepLearning.ai launched a new short course [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen).

🎉 **May 24, 2024:** Foundation Capital published an article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and a video [AI in the Real World Episode 2: Exploring Multi-Agent AI and AutoGen with Chi Wang](https://www.youtube.com/watch?v=RLwyXRVvlNk).

🎉 **April 17, 2024:** Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and [What's next for AI agentic workflows](https://youtu.be/sal78ACtGTc?si=JduUzN_1kDnMq0vF) at Sequoia Capital's AI Ascent (Mar 26).

[More Announcements](announcements.md)

## Code Style and Linting

This project uses pre-commit hooks to ensure code quality. Before contributing:

1.  Install pre-commit:

```bash
pip install pre-commit
pre-commit install
```

2.  The hooks run automatically on commit, or run them manually:

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

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" alt="Contributors">
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

For more details, please see the [NOTICE](./NOTICE.md) file.