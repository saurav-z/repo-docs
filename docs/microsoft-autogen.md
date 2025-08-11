<div align="center">
<img src="https://microsoft.github.io/autogen/0.2/img/ag.svg" alt="AutoGen Logo" width="100">

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40pyautogen)](https://twitter.com/pyautogen)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Company?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/105812540)
[![Discord](https://img.shields.io/badge/discord-chat-green?logo=discord)](https://aka.ms/autogen-discord)
[![Documentation](https://img.shields.io/badge/Documentation-AutoGen-blue?logo=read-the-docs)](https://microsoft.github.io/autogen/)
[![Blog](https://img.shields.io/badge/Blog-AutoGen-blue?logo=blogger)](https://devblogs.microsoft.com/autogen/)

</div>

# AutoGen: Build Powerful Multi-Agent AI Applications

**AutoGen empowers developers to build advanced AI applications with autonomous or human-in-the-loop multi-agent workflows.**

[**Explore the AutoGen Repository**](https://github.com/microsoft/autogen)

## Key Features

*   **Multi-Agent Framework:** Design and orchestrate complex AI agent interactions.
*   **Flexible Architecture:** Utilize a layered and extensible design, suitable for both high-level and low-level development.
*   **Rapid Prototyping:** Quickly build and test agent-based solutions with the AgentChat API.
*   **Extensible Ecosystem:** Leverage a growing library of extensions for LLM clients, code execution, and more.
*   **No-Code GUI:** Prototype and run multi-agent workflows with AutoGen Studio.
*   **Comprehensive Tools:** Access tools for benchmarking, debugging, and deployment.
*   **.NET and Python Support:** Build cross-language applications.

## Installation

AutoGen requires Python 3.10 or later.

```bash
# Install AgentChat and OpenAI client
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

For AutoGen Studio:

```bash
pip install -U "autogenstudio"
```

**Note:** If upgrading from AutoGen v0.2, consult the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html) for important update instructions.

## Quickstart

### Hello World

Create a basic assistant agent using OpenAI's GPT-4o model. See [other supported models](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/models.html).

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

Create a web browsing assistant agent using the Playwright MCP server.

```python
# First run `npm install -g @playwright/mcp@latest` to install the MCP server.
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMessageTermination
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
        team = RoundRobinGroupChat(
            [agent],
            termination_condition=TextMessageTermination(source="web_browsing_assistant"),
        )
        await Console(team.run_stream(task="Find out how many contributors for the microsoft/autogen repository"))


asyncio.run(main())
```

> **Warning**: Only connect to trusted MCP servers as they may execute commands
> in your local environment or expose sensitive information.

### AutoGen Studio

Easily prototype and run multi-agent workflows without writing code using AutoGen Studio.

```bash
# Run AutoGen Studio on http://localhost:8080
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Choose AutoGen?

<div align="center">
  <img src="autogen-landing.jpg" alt="AutoGen Landing" width="500">
</div>

AutoGen offers a complete ecosystem for building AI agents, especially multi-agent systems, including a robust framework, developer tools, and practical applications.

The AutoGen _framework_ features a layered design that prioritizes flexibility and extensibility. This design allows you to utilize the framework at different levels of abstraction:

*   **Core API:** Implements message passing, event-driven agents, and local/distributed runtime. Supports cross-language for .NET and Python.
*   **AgentChat API:** Provides an opinionated and simpler API for fast prototyping, built on the Core API. Supports common multi-agent patterns.
*   **Extensions API:** Enables first- and third-party extensions that expand framework capabilities, including LLM clients (e.g., OpenAI) and code execution.

AutoGen also provides essential _developer tools_:

<div align="center">
  <img src="https://media.githubusercontent.com/media/microsoft/autogen/refs/heads/main/python/packages/autogen-studio/docs/ags_screen.png" alt="AutoGen Studio Screenshot" width="500">
</div>

*   **AutoGen Studio:** A no-code GUI for creating multi-agent applications.
*   **AutoGen Bench:** A benchmarking suite for evaluating agent performance.

Build applications tailored to your needs using the AutoGen framework and developer tools. For example, [Magentic-One](./python/packages/magentic-one-cli/) showcases a state-of-the-art multi-agent team built using AgentChat API and Extensions API, designed to handle tasks that require web browsing, code execution, and file handling.

## Get Involved

Join the thriving AutoGen community. We offer weekly office hours and talks with maintainers and community members. Connect with us via:

*   [Discord server](https://aka.ms/autogen-discord) for real-time chat
*   [GitHub Discussions](https://github.com/microsoft/autogen/discussions) for Q&A
*   [Blog](https://devblogs.microsoft.com/autogen/) for tutorials and updates

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

We welcome contributions!  Consult [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## FAQ

Find answers to common questions in our [FAQ](./FAQ.md).

## Legal Notices

Microsoft and contributors license the content under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode), and any code in the repository under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) and [LICENSE-CODE](LICENSE-CODE) files.

Microsoft, Windows, Azure, and other referenced products/services are trademarks of Microsoft.  The licenses do not grant rights to use Microsoft's names, logos, or trademarks.

Privacy information: <https://go.microsoft.com/fwlink/?LinkId=521839>

Microsoft and contributors reserve all other rights.

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
```
Key improvements and SEO optimization:

*   **Clear and Concise Hook:** The introductory sentence immediately highlights the core value proposition.
*   **Keyword Optimization:** Includes relevant keywords like "Multi-Agent AI," "AI Applications," "Autonomous Agents," and "AutoGen" throughout the headings and text.
*   **Structured Headings:** Uses clear, descriptive headings (Key Features, Installation, Quickstart, Why Choose AutoGen?, Get Involved, Where to Go Next?, Contributing, FAQ, Legal Notices) for improved readability and SEO.
*   **Bulleted Key Features:** Highlights the most important aspects of the project using bullet points, making it easy for users to understand the value.
*   **Internal Links:** Includes links within the document (e.g., to the FAQ, CONTRIBUTING.md, and various sections) to improve navigation and user experience.
*   **External Links:** Properly formatted and easy to read.
*   **Concise Language:** Simplifies and clarifies the original text for better comprehension.
*   **Call to Action:** Encourages users to get involved and contribute to the project.
*   **Focus on Benefits:**  The "Why Choose AutoGen?" section emphasizes the benefits of using the framework.
*   **Alt Text for Images:** Added alt text for all images.
*   **Back to Top Links:** Maintain navigation within the README for easier use.