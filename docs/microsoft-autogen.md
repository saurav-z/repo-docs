<div align="center">
<img src="https://microsoft.github.io/autogen/0.2/img/ag.svg" alt="AutoGen Logo" width="100">

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40pyautogen)](https://twitter.com/pyautogen)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Company?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/105812540)
[![Discord](https://img.shields.io/badge/discord-chat-green?logo=discord)](https://aka.ms/autogen-discord)
[![Documentation](https://img.shields.io/badge/Documentation-AutoGen-blue?logo=read-the-docs)](https://microsoft.github.io/autogen/)
[![Blog](https://img.shields.io/badge/Blog-AutoGen-blue?logo=blogger)](https://devblogs.microsoft.com/autogen/)
</div>

# AutoGen: Build Multi-Agent AI Applications with Ease

**AutoGen is a powerful framework that empowers developers to create cutting-edge multi-agent AI applications, whether they're designed to act autonomously or collaboratively with humans.** ([Original Repo](https://github.com/microsoft/autogen))

## Key Features & Benefits

*   **Multi-Agent Framework:** Develop complex workflows and interactions between AI agents.
*   **Flexible Architecture:** Utilize a layered design with Core, AgentChat, and Extensions APIs for customization.
*   **Rapid Prototyping:** The AgentChat API simplifies the creation of multi-agent systems.
*   **Extensible:** Integrate with various LLM clients (OpenAI, AzureOpenAI, etc.) and custom extensions.
*   **No-Code GUI:** AutoGen Studio provides a user-friendly interface for designing and running multi-agent applications.
*   **Benchmarking Tools:** Evaluate agent performance using the AutoGen Bench suite.
*   **Active Community:** Benefit from weekly office hours, Discord support, and a blog for updates and tutorials.

## Getting Started

### Installation

AutoGen requires **Python 3.10 or later**.

```bash
# Install AgentChat and OpenAI client from Extensions
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

The current stable version can be found in the [releases](https://github.com/microsoft/autogen/releases). If you are upgrading from AutoGen v0.2, please refer to the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html) for detailed instructions on how to update your code and configurations.

```bash
# Install AutoGen Studio for no-code GUI
pip install -U "autogenstudio"
```

### Quickstart Examples

#### Hello World

Create a simple assistant agent:

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4.1")
    agent = AssistantAgent("assistant", model_client=model_client)
    print(await agent.run(task="Say 'Hello World!'"))
    await model_client.close()

asyncio.run(main())
```

#### MCP Server

Create a web browsing assistant agent:

```python
# First run `npm install -g @playwright/mcp@latest` to install the MCP server.
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams


async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4.1")
    server_params = StdioServerParams(
        command="npx",
        args=[
            "@playwright/mcp@latest",
            "--headless",
        ],
    )
    async with McpWorkbench(server_params) as mcp:
        agent = AssistantAgent(
            "web_browsing_assistant",
            model_client=model_client,
            workbench=mcp, # For multiple MCP servers, put them in a list.
            model_client_stream=True,
            max_tool_iterations=10,
        )
        await Console(agent.run_stream(task="Find out how many contributors for the microsoft/autogen repository"))


asyncio.run(main())
```

> **Warning:** Only connect to trusted MCP servers as they may execute commands in your local environment or expose sensitive information.

#### Multi-Agent Orchestration

Set up a basic multi-agent system:

```python
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.tools import AgentTool
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4.1")

    math_agent = AssistantAgent(
        "math_expert",
        model_client=model_client,
        system_message="You are a math expert.",
        description="A math expert assistant.",
        model_client_stream=True,
    )
    math_agent_tool = AgentTool(math_agent, return_value_as_last_message=True)

    chemistry_agent = AssistantAgent(
        "chemistry_expert",
        model_client=model_client,
        system_message="You are a chemistry expert.",
        description="A chemistry expert assistant.",
        model_client_stream=True,
    )
    chemistry_agent_tool = AgentTool(chemistry_agent, return_value_as_last_message=True)

    agent = AssistantAgent(
        "assistant",
        system_message="You are a general assistant. Use expert tools when needed.",
        model_client=model_client,
        model_client_stream=True,
        tools=[math_agent_tool, chemistry_agent_tool],
        max_tool_iterations=10,
    )
    await Console(agent.run_stream(task="What is the integral of x^2?"))
    await Console(agent.run_stream(task="What is the molecular weight of water?"))


asyncio.run(main())
```

Explore the [AgentChat documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html) for advanced multi-agent orchestration.

### AutoGen Studio

Prototype and run workflows using a no-code GUI:

```bash
# Run AutoGen Studio on http://localhost:8080
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Choose AutoGen?

AutoGen provides a complete ecosystem for creating and deploying AI agents, especially multi-agent workflows, with:

*   **Flexible Framework:** Leverage a layered and extensible design to suit your needs.
*   **Developer Tools:** Utilize AutoGen Studio for no-code development and AutoGen Bench for performance evaluation.
*   **Application Building:** Easily create applications for your specific domain, such as the Magentic-One CLI.
*   **Community Support:** Join a thriving community with office hours, Discord, and a blog.

## Where to Go Next

Explore the following resources to get started:

| Resource          | Description                                     | Link                                                                                                                                 |
| ----------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Python**        | Core Python API                                 | [![Python](https://img.shields.io/badge/AutoGen-Python-blue?logo=python&logoColor=white)](./python)                                  |
| **.NET**          | .NET API                                        | [![.NET](https://img.shields.io/badge/AutoGen-.NET-green?logo=.net&logoColor=white)](./dotnet)                                     |
| **AutoGen Studio** | No-Code GUI for Building Multi-Agent Apps | [![Studio](https://img.shields.io/badge/AutoGen-Studio-purple?logo=visual-studio&logoColor=white)](./python/packages/autogen-studio) |
| **Installation**  | Installation Guide                              | [![Installation](https://img.shields.io/badge/Install-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html)          |
| **Quickstart**    | Quickstart Guide                                | [![Quickstart](https://img.shields.io/badge/Quickstart-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html#)           |
| **Tutorial**      | Tutorials                                       | [![Tutorial](https://img.shields.io/badge/Tutorial-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/index.html)             |
| **API Reference** | API Documentation                                | [![API](https://img.shields.io/badge/Docs-blue)](https://microsoft.github.io/autogen/stable/reference/index.html#)                                            |
| **Packages**      | Python Packages (PyPi)                           | [![PyPi autogen-core](https://img.shields.io/badge/PyPi-autogen--core-blue?logo=pypi)](https://pypi.org/project/autogen-core/) <br> [![PyPi autogen-agentchat](https://img.shields.io/badge/PyPi-autogen--agentchat-blue?logo=pypi)](https://pypi.org/project/autogen-agentchat/) <br> [![PyPi autogen-ext](https://img.shields.io/badge/PyPi-autogen--ext-blue?logo=pypi)](https://pypi.org/project/autogen-ext/) |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines. Join our community and help improve AutoGen.

## Support & Community

*   **FAQ:** [Frequently Asked Questions (FAQ)](./FAQ.md)
*   **Discussions:** [GitHub Discussions](https://github.com/microsoft/autogen/discussions)
*   **Discord:** [Discord Server](https://aka.ms/autogen-discord)
*   **Blog:** [AutoGen Blog](https://devblogs.microsoft.com/autogen/)

## Legal Notices

See the [LICENSE](LICENSE) and [LICENSE-CODE](LICENSE-CODE) files for licensing information. Refer to the Microsoft trademarks guidelines and privacy information as referenced in the original README.

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>