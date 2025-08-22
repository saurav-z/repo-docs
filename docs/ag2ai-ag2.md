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

# AG2: Build Powerful AI Agent Systems with Open-Source Framework

**AG2 is an open-source agent orchestration framework, empowering developers to create and collaborate with AI agents to solve complex tasks.** This project is a fork of AutoGen and is maintained by a [dynamic group of volunteers](MAINTAINERS.md).

[Explore the original repo on GitHub](https://github.com/ag2ai/ag2).

## Key Features

*   **Multi-Agent Collaboration:** Design agents that interact seamlessly with each other for advanced problem-solving.
*   **LLM & Tool Integration:** Leverage various Large Language Models (LLMs) and integrate tool use for enhanced functionality.
*   **Flexible Workflows:** Implement autonomous and human-in-the-loop workflows to suit your project needs.
*   **Conversation Patterns:** Utilize pre-built multi-agent conversation patterns for efficient task management.
*   **Extensible Architecture:** Easily adapt the framework with custom agents, tools, and orchestration patterns.

## Core Concepts

*   **Conversable Agent**: Enables direct communication between AI entities.
*   **Human-in-the-Loop**: Integrates human input for oversight and critical decision-making.
*   **Orchestration**: Supports various multi-agent patterns, including GroupChat and Swarm.
*   **Tools**: Allows agents to utilize external APIs, data, and functionalities.
*   **Advanced Agentic Design Patterns**: Implement structured outputs, RAG, code execution, and more.

## Getting Started

For a detailed guide, please refer to the [Basic Concepts](https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/installing-ag2/) section in our documentation.

### Installation

Install AG2 with Python >= 3.10, < 3.14 via pip:

```bash
pip install ag2[openai]
```

### Setup API Keys

Configure your LLM keys using the `OAI_CONFIG_LIST` file.  See `OAI_CONFIG_LIST_sample` for an example.

```json
[
  {
    "model": "gpt-4o",
    "api_key": "<your OpenAI API key here>"
  }
]
```

### Run Your First Agent

Test your setup by running a simple agent:

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

with llm_config:
    assistant = AssistantAgent("assistant")
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
```

## Example Applications

Explore a wide range of practical applications:

*   [Build with AG2](https://github.com/ag2ai/build-with-ag2)
*   [Jupyter Notebooks](notebook)

## Announcements

*   **Nov 11, 2024:** AutoGen is evolving into **AG2**! [AG2AI](https://github.com/ag2ai) hosts development with open governance, and you can see [AG2's new look](https://ag2.ai/).
*   **License:** Now using the Apache 2.0 license from v0.3 for open-source collaboration.
*   **May 29, 2024:** DeepLearning.ai launched a course: [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen).
*   **May 24, 2024:** Foundation Capital article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97).
*   **Apr 17, 2024:** Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and [What's next for AI agentic workflows](https://youtu.be/sal78ACtGTc?si=JduUzN_1kDnMq0vF) at Sequoia Capital's AI Ascent (Mar 26).

[More Announcements](announcements.md)

## Contributors

<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

## Code Style & Linting

Follow these steps to ensure code quality:

1.  Install pre-commit:

```bash
pip install pre-commit
pre-commit install
```

2.  Run hooks before committing, or manually:

```bash
pre-commit run --all-files
```

## Related Papers

*   [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
*   And other relevant papers...

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

It is a spin-off of [AutoGen](https://github.com/microsoft/autogen).

*   Original AutoGen code is licensed under the MIT License (see [LICENSE\_original\_MIT](./license_original/LICENSE_original_MIT)).
*   AG2 modifications are under the Apache License, Version 2.0 (see [LICENSE](./LICENSE)).

See [NOTICE](./NOTICE.md) for details.