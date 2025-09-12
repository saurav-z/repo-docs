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

# AG2: Open-Source AgentOS for AI Agents

**AG2 empowers developers to build, experiment, and deploy AI agents with a versatile and collaborative framework.**  Originally evolved from AutoGen, AG2 is a fully open-source platform designed to streamline the development and research of agentic AI.  It offers a robust set of features, including multi-agent communication, LLM and tool integration, and customizable workflows, all contributing to the creation of sophisticated AI solutions.

➡️ **[Explore the AG2 Repository on GitHub](https://github.com/ag2ai/ag2)**

## Key Features

*   **Multi-Agent Communication:** Facilitates seamless interaction and collaboration between multiple AI agents.
*   **LLM and Tool Integration:**  Supports integration with various Large Language Models (LLMs) and external tools for enhanced capabilities.
*   **Flexible Workflows:** Enables both autonomous and human-in-the-loop workflows, providing flexibility in agent design.
*   **Conversation Patterns:** Offers pre-built multi-agent conversation patterns (e.g., swarms, group chats) to simplify complex interactions.
*   **Extensible Architecture:**  Designed for extensibility, allowing for custom agent behaviors and interactions.

## Core Concepts

### Conversable Agent

The fundamental building block of AG2, designed to enable seamless communication between AI entities. This core agent type handles message exchange and response generation, serving as the base class for all agents in the framework.

In the example below, we'll create a simple information validation workflow with two specialized agents that communicate with each other:

Note: Before running this code, make sure to set your `OPENAI_API_KEY` as an environment variable. This example uses `gpt-4o-mini`, but you can replace it with any other [model](https://docs.ag2.ai/latest/docs/user-guide/models/amazon-bedrock) supported by AG2.

```python
# 1. Import ConversableAgent class
from autogen import ConversableAgent, LLMConfig

# 2. Define our LLM configuration for OpenAI's GPT-4o mini
#    uses the OPENAI_API_KEY environment variable
llm_config = LLMConfig({
    "api_type": "openai",
    "model": "gpt-5-mini",
})

# 3. Create our LLM agent
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

# 4. Start the conversation
assistant.initiate_chat(
    recipient=fact_checker,
    message="What is AG2?",
    max_turns=2
)
```

### Human in the loop

Human oversight is crucial for many AI workflows, especially when dealing with critical decisions, creative tasks, or situations requiring expert judgment. AG2 makes integrating human feedback seamless through its human-in-the-loop functionality.
You can configure how and when human input is solicited using the `human_input_mode` parameter:

- `ALWAYS`: Requires human input for every response
- `NEVER`: Operates autonomously without human involvement
- `TERMINATE`: Only requests human input to end conversations

For convenience, AG2 provides the specialized `UserProxyAgent` class that automatically sets `human_input_mode` to `ALWAYS` and supports code execution:

Note: Before running this code, make sure to set your `OPENAI_API_KEY` as an environment variable. This example uses `gpt-4o-mini`, but you can replace it with any other [model](https://docs.ag2.ai/latest/docs/user-guide/models/amazon-bedrock) supported by AG2.

```python
# 1. Import ConversableAgent and UserProxyAgent classes
from autogen import ConversableAgent, UserProxyAgent, LLMConfig

# 2. Define our LLM configuration for OpenAI's GPT-4o mini
#    uses the OPENAI_API_KEY environment variable
llm_config = LLMConfig({
    "api_type": "openai",
    "model": "gpt-5-mini",
})

# 3. Create our LLM agent
assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

# 4. Create a human agent with manual input mode
human = ConversableAgent(
    name="human",
    human_input_mode="ALWAYS"
)

# or
human = UserProxyAgent(
    name="human",
    code_execution_config={"work_dir": "coding", "use_docker": False},
)

# 5. Start the chat
human.initiate_chat(
    recipient=assistant,
    message="Hello! What's 2 + 2?"
)

```

### Orchestrating multiple agents

AG2 enables sophisticated multi-agent collaboration through flexible orchestration patterns, allowing you to create dynamic systems where specialized agents work together to solve complex problems.

The framework offers both custom orchestration and several built-in collaboration patterns including `GroupChat` and `Swarm`.

Here's how to implement a collaborative team for curriculum development using GroupChat:

Note: Before running this code, make sure to set your `OPENAI_API_KEY` as an environment variable. This example uses `gpt-4o-mini`, but you can replace it with any other [model](https://docs.ag2.ai/latest/docs/user-guide/models/amazon-bedrock) supported by AG2.

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

# Put your key in the OPENAI_API_KEY environment variable
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

# 1. Add a separate 'description' for our planner and reviewer agents
planner_description = "Creates or revises lesson plans."

reviewer_description = """Provides one round of reviews to a lesson plan
for the lesson_planner to revise."""

lesson_planner = ConversableAgent(
    name="planner_agent",
    system_message=planner_message,
    description=planner_description,
    llm_config=llm_config,
)

lesson_reviewer = ConversableAgent(
    name="reviewer_agent",
    system_message=reviewer_message,
    description=reviewer_description,
    llm_config=llm_config,
)

# 2. The teacher's system message can also be used as a description, so we don't define it
teacher_message = """You are a classroom teacher.
You decide topics for lessons and work with a lesson planner.
and reviewer to create and finalise lesson plans.
When you are happy with a lesson plan, output "DONE!".
"""

teacher = ConversableAgent(
    name="teacher_agent",
    system_message=teacher_message,
    # 3. Our teacher can end the conversation by saying DONE!
    is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),
    llm_config=llm_config,
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

When executed, this code creates a collaborative system where the teacher initiates the conversation, and the lesson planner and reviewer agents work together to create and refine a lesson plan. The GroupChatManager orchestrates the conversation, selecting the next agent to respond based on the context of the discussion.

For workflows requiring more structured processes, explore the Group Chat pattern in the detailed [documentation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/group-chat/introduction).

### Tools

Agents gain significant utility through tools as they provide access to external data, APIs, and functionality.

Note: Before running this code, make sure to set your `OPENAI_API_KEY` as an environment variable. This example uses `gpt-4o-mini`, but you can replace it with any other [model](https://docs.ag2.ai/latest/docs/user-guide/models/amazon-bedrock) supported by AG2.

```python
from datetime import datetime
from typing import Annotated

from autogen import ConversableAgent, register_function, LLMConfig

# Put your key in the OPENAI_API_KEY environment variable
llm_config = LLMConfig({
    "api_type": "openai",
    "model": "gpt-5-mini",
})

# 1. Our tool, returns the day of the week for a given date
def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")

# 2. Agent for determining whether to run the tool
date_agent = ConversableAgent(
    name="date_agent",
    system_message="You get the day of the week for a given date.",
    llm_config=llm_config,
)

# 3. And an agent for executing the tool
executor_agent = ConversableAgent(
    name="executor_agent",
    human_input_mode="NEVER",
    llm_config=llm_config,
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

### Advanced agentic design patterns

AG2 supports more advanced concepts to help you build your AI agent workflows. You can find more information in the documentation.

- [Structured Output](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/structured-outputs)
- [Ending a conversation](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/orchestration/ending-a-chat/)
- [Retrieval Augmented Generation (RAG)](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/rag/)
- [Code Execution](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/code-execution)
- [Tools with Secrets](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/tools-with-secrets/)

## Getting Started

AG2 provides a flexible and intuitive setup. For a step-by-step walk through of AG2 concepts and code, see [Basic Concepts](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/installing-ag2/) in our documentation.

### Installation

AG2 requires **Python version >= 3.10, < 3.14**. AG2 is available via `ag2` (or its alias `autogen`) on PyPI.

```bash
pip install ag2[openai]
```

Minimal dependencies are installed by default. You can install extra options based on the features you need.

### Setup your API keys

To keep your LLM dependencies neat we recommend using the `OAI_CONFIG_LIST` file to store your API keys.

You can use the sample file `OAI_CONFIG_LIST_sample` as a template.

```json
[
  {
    "model": "gpt-5",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run your first agent

Create a script or a Jupyter Notebook and run your first agent.

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

assistant = AssistantAgent("assistant", llm_config=llm_config)

user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})

user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
# This initiates an automated chat between the two agents to solve the task
```

## Example Applications

Explore a wide range of practical applications and use cases with AG2. For a collection of jupyter notebooks as a starting point, check out our [Jupyter Notebooks](notebook).

*   [Build with AG2](https://github.com/ag2ai/build-with-ag2)

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

Ensure code quality and consistency by adhering to the following guidelines. This project uses pre-commit hooks.

1.  Install pre-commit:

    ```bash
    pip install pre-commit
    pre-commit install
    ```

2.  The hooks will run automatically on commit, or you can run them manually:

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

If you use AG2 in your research, please cite us:

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