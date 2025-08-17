<div align="center">
  <img src="https://microsoft.github.io/autogen/0.2/img/ag.svg" alt="AutoGen Logo" width="100">

  [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40pyautogen)](https://twitter.com/pyautogen)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Company?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/105812540)
  [![Discord](https://img.shields.io/badge/discord-chat-green?logo=discord)](https://aka.ms/autogen-discord)
  [![Documentation](https://img.shields.io/badge/Documentation-AutoGen-blue?logo=read-the-docs)](https://microsoft.github.io/autogen/)
  [![Blog](https://img.shields.io/badge/Blog-AutoGen-blue?logo=blogger)](https://devblogs.microsoft.com/autogen/)
</div>

# AutoGen: Build Powerful Multi-Agent AI Applications

**AutoGen** is a versatile framework that empowers developers to create cutting-edge multi-agent AI applications, fostering autonomous workflows and human-AI collaboration.  [Explore the AutoGen GitHub repository](https://github.com/microsoft/autogen).

## Key Features

*   **Multi-Agent Workflows:** Design and implement complex AI workflows involving multiple agents.
*   **Extensible Architecture:** Build on a layered and flexible design, enabling use from high-level APIs to low-level components.
*   **Comprehensive API:** Utilize Core API, AgentChat API, and Extensions API for a range of use cases.
*   **Developer Tools:** Leverage AutoGen Studio (no-code GUI) and AutoGen Bench (benchmarking suite) for streamlined development.
*   **Cross-Language Support:** Support for both Python and .NET.
*   **Community Driven:** Join a vibrant ecosystem with weekly office hours, a Discord server, GitHub Discussions, and a blog.

## Installation

AutoGen requires **Python 3.10 or later**.

```bash
# Install AgentChat and OpenAI client from Extensions
pip install -U "autogen-agentchat" "autogen-ext[openai]"

# Install AutoGen Studio for no-code GUI
pip install -U "autogenstudio"
```

If you are upgrading from AutoGen v0.2, please refer to the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html) for instructions on how to update your code and configurations.

## Quickstart

### Hello World

Create an assistant agent using OpenAI's GPT-4o model.

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    agent = AssistantAgent("assistant", model_client=model_client)
    print(await agent.run(task="Say 'Hello World!'"))
    await model_client.close()

asyncio.run(main())
```

### MCP Server

Create a web browsing assistant agent that uses the Playwright MCP server.

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

> **Warning**: Only connect to trusted MCP servers as they may execute commands
> in your local environment or expose sensitive information.

### AutoGen Studio

Use AutoGen Studio to prototype and run multi-agent workflows without writing code.

```bash
# Run AutoGen Studio on http://localhost:8080
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Choose AutoGen?

<div align="center">
  <img src="autogen-landing.jpg" alt="AutoGen Landing" width="500">
</div>

AutoGen provides a comprehensive ecosystem, including a powerful framework, essential developer tools, and real-world applications, making it easy to create AI agents and multi-agent workflows.  AutoGen’s architecture allows you to build at different levels of abstraction.

*   **Framework:**
    *   **Core API:** Implements message passing, event-driven agents, and local/distributed runtimes. Supports cross-language support for .NET and Python.
    *   **AgentChat API:** Provides a simpler API for rapid prototyping, built on top of the Core API.
    *   **Extensions API:** Enables first- and third-party extensions to expand framework capabilities.
*   **Developer Tools:**
    *   **AutoGen Studio:** A no-code GUI for building multi-agent applications.
    *   **AutoGen Bench:** A benchmarking suite for evaluating agent performance.

Use AutoGen to create applications in your domain. An example is [Magentic-One](https://github.com/microsoft/autogen/tree/main/python/packages/magentic-one-cli/), a multi-agent team built using AgentChat API and Extensions API.

Join a thriving community with weekly office hours and talks. We have a [Discord server](https://aka.ms/autogen-discord) for real-time chat, GitHub Discussions for Q&A, and a blog for tutorials and updates.

## Where to Go Next?

<div align="center">

|               | [![Python](https://img.shields.io/badge/AutoGen-Python-blue?logo=python&logoColor=white)](./python)                                                                                                                                                                                                                                                                                                                | [![.NET](https://img.shields.io/badge/AutoGen-.NET-green?logo=.net&logoColor=white)](./dotnet)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | [![Studio](https://img.shields.io/badge/AutoGen-Studio-purple?logo=visual-studio&logoColor=white)](./python/packages/autogen-studio)                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Installation  | [![Installation](https://img.shields.io/badge/Install-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html)                                                                                                                                                                                                                                                         | [![Install](https://img.shields.io/badge/Install-green)](https://microsoft.github.io/autogen/dotnet/dev/core/installation.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | [![Install](https://img.shields.io/badge/Install-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/installation.html) |
| Quickstart    | [![Quickstart](https://img.shields.io/badge/Quickstart-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html#)                                                                                                                                                                                                                                                         | [![Quickstart](https://img.shields.io/badge/Quickstart-green)](https://microsoft.github.io/autogen/dotnet/dev/core/index.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | [![Usage](https://img.shields.io/badge/Quickstart-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)      |
| Tutorial      | [![Tutorial](https://img.shields.io/badge/Tutorial-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/index.html)                                                                                                                                                                                                                                                          | [![Tutorial](https://img.shields.io/badge/Tutorial-green)](https://microsoft.github.io/autogen/dotnet/dev/core/tutorial.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | [![Usage](https://img.shields.io/badge/Tutorial-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| API Reference | [![API](https://img.shields.io/badge/Docs-blue)](https://microsoft.github.io/autogen/stable/reference/index.html#)                                                                                                                                                                                                                                                                                                 | [![API](https://img.shields.io/badge/Docs-green)](https://microsoft.github.io/autogen/dotnet/dev/api/Microsoft.AutoGen.Contracts.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | [![API](https://img.shields.io/badge/Docs-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html)               |
| Packages      | [![PyPi autogen-core](https://img.shields.io/badge/PyPi-autogen--core-blue?logo=pypi)](https://pypi.org/project/autogen-core/) <br> [![PyPi autogen-agentchat](https://img.shields.io/badge/PyPi-autogen--agentchat-blue?logo=pypi)](https://pypi.org/project/autogen-agentchat/) <br> [![PyPi autogen-ext](https://img.shields.io/badge/PyPi-autogen--ext-blue?logo=pypi)](https://pypi.org/project/autogen-ext/) | [![NuGet Contracts](https://img.shields.io/badge/NuGet-Contracts-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Contracts/) <br> [![NuGet Core](https://img.shields.io/badge/NuGet-Core-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core/) <br> [![NuGet Core.Grpc](https://img.shields.io/badge/NuGet-Core.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core.Grpc/) <br> [![NuGet RuntimeGateway.Grpc](https://img.shields.io/badge/NuGet-RuntimeGateway.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.RuntimeGateway.Grpc/) | [![PyPi autogenstudio](https://img.shields.io/badge/PyPi-autogenstudio-purple?logo=pypi)](https://pypi.org/project/autogenstudio/)                          |

</div>

## Contributing

Contribute to the project by reviewing the [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines. Join our community and help us make AutoGen better!

## Support and Community

*   [Frequently Asked Questions (FAQ)](./FAQ.md)
*   [GitHub Discussions](https://github.com/microsoft/autogen/discussions)
*   [Discord server](https://aka.ms/autogen-discord)
*   [Blog](https://devblogs.microsoft.com/autogen/)

## Legal Notices

(Content from original README)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>