<div align="center">
<img src="https://microsoft.github.io/autogen/0.2/img/ag.svg" alt="AutoGen Logo" width="100">

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40pyautogen)](https://twitter.com/pyautogen)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Company?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/105812540)
[![Discord](https://img.shields.io/badge/discord-chat-green?logo=discord)](https://aka.ms/autogen-discord)
[![Documentation](https://img.shields.io/badge/Documentation-AutoGen-blue?logo=read-the-docs)](https://microsoft.github.io/autogen/)
[![Blog](https://img.shields.io/badge/Blog-AutoGen-blue?logo=blogger)](https://devblogs.microsoft.com/autogen/)

</div>

# AutoGen: Build Powerful Multi-Agent AI Applications

AutoGen is a versatile framework for creating and managing sophisticated multi-agent AI systems, empowering developers to build autonomous and collaborative AI applications.  [Explore the original repository](https://github.com/microsoft/autogen).

## Key Features

*   **Multi-Agent Workflows:** Design and implement complex AI workflows involving multiple agents that can collaborate, communicate, and execute tasks.
*   **Extensible Architecture:** Leverage a layered and modular design, allowing for flexibility in development, from high-level APIs to low-level customization.
*   **AgentChat API:** Quickly prototype and build multi-agent applications with an intuitive API.
*   **AutoGen Studio:** Utilize a no-code GUI for designing, testing, and deploying multi-agent workflows, simplifying the development process.
*   **OpenAI & Azure OpenAI Integration:** Seamlessly integrate with leading LLM providers to power your agents.
*   **Comprehensive Tools:** Benefit from a rich ecosystem including the AutoGen Bench for performance evaluation.
*   **.NET Support:** Develop applications in .NET environments.
*   **Community & Support:** Join a thriving community with active discussions, tutorials, and a Discord server for support.

## Installation

AutoGen requires Python 3.10 or later.

```bash
# Install AgentChat and OpenAI client from Extensions
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

The current stable version is v0.4.  If upgrading from AutoGen v0.2, see the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html).

```bash
# Install AutoGen Studio for no-code GUI
pip install -U "autogenstudio"
```

## Quickstart Examples

### Hello World

Create a basic assistant agent using OpenAI's GPT-4o model.

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

Create a web browsing assistant using the Playwright MCP server.

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

> **Warning**: Connect to trusted MCP servers only.

### AutoGen Studio

Launch AutoGen Studio to prototype and run multi-agent workflows without coding.

```bash
# Run AutoGen Studio on http://localhost:8080
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Use AutoGen?

<div align="center">
  <img src="autogen-landing.jpg" alt="AutoGen Landing" width="500">
</div>

AutoGen empowers you to create advanced AI agents and multi-agent workflows with a flexible framework, developer tools, and a thriving ecosystem.

The framework offers a layered architecture:

*   **Core API:** Provides the foundation for message passing, event-driven agents, and flexible runtimes, with cross-language support for .NET and Python.
*   **AgentChat API:** Simplifies rapid prototyping with an opinionated API built on the Core API, supporting common multi-agent patterns.
*   **Extensions API:** Enables expanding capabilities through first- and third-party extensions, like LLM clients (OpenAI, AzureOpenAI) and code execution features.

The ecosystem includes these developer tools:

<div align="center">
  <img src="https://media.githubusercontent.com/media/microsoft/autogen/refs/heads/main/python/packages/autogen-studio/docs/ags_screen.png" alt="AutoGen Studio Screenshot" width="500">
</div>

*   **AutoGen Studio:**  A no-code GUI for building multi-agent applications.
*   **AutoGen Bench:**  A benchmarking suite for evaluating agent performance.

Build applications using AutoGen – for example, [Magentic-One](https://github.com/microsoft/autogen/tree/main/python/packages/magentic-one-cli) is a state-of-the-art multi-agent team.

Join the AutoGen community with weekly office hours, Discord for real-time chat, and GitHub Discussions for Q&A.

## Getting Started

<div align="center">

|               | [![Python](https://img.shields.io/badge/AutoGen-Python-blue?logo=python&logoColor=white)](./python)                                                                                                                                                                                                                                                                                                                | [![.NET](https://img.shields.io/badge/AutoGen-.NET-green?logo=.net&logoColor=white)](./dotnet)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | [![Studio](https://img.shields.io/badge/AutoGen-Studio-purple?logo=visual-studio&logoColor=white)](./python/packages/autogen-studio)                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Installation  | [![Installation](https://img.shields.io/badge/Install-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html)                                                                                                                                                                                                                                                         | [![Install](https://img.shields.io/badge/Install-green)](https://microsoft.github.io/autogen/dotnet/dev/core/installation.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | [![Install](https://img.shields.io/badge/Install-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/installation.html) |
| Quickstart    | [![Quickstart](https://img.shields.io/badge/Quickstart-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html#)                                                                                                                                                                                                                                                         | [![Quickstart](https://img.shields.io/badge/Quickstart-green)](https://microsoft.github.io/autogen/dotnet/dev/core/index.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | [![Usage](https://img.shields.io/badge/Quickstart-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)      |
| Tutorial      | [![Tutorial](https://img.shields.io/badge/Tutorial-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/index.html)                                                                                                                                                                                                                                                          | [![Tutorial](https://img.shields.io/badge/Tutorial-green)](https://microsoft.github.io/autogen/dotnet/dev/core/tutorial.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | [![Usage](https://img.shields.io/badge/Tutorial-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| API Reference | [![API](https://img.shields.io/badge/Docs-blue)](https://microsoft.github.io/autogen/stable/reference/index.html#)                                                                                                                                                                                                                                                                                                 | [![API](https://img.shields.io/badge/Docs-green)](https://microsoft.github.io/autogen/dotnet/dev/api/Microsoft.AutoGen.Contracts.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | [![API](https://img.shields.io/badge/Docs-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html)               |
| Packages      | [![PyPi autogen-core](https://img.shields.io/badge/PyPi-autogen--core-blue?logo=pypi)](https://pypi.org/project/autogen-core/) <br> [![PyPi autogen-agentchat](https://img.shields.io/badge/PyPi-autogen--agentchat-blue?logo=pypi)](https://pypi.org/project/autogen-agentchat/) <br> [![PyPi autogen-ext](https://img.shields.io/badge/PyPi-autogen--ext-blue?logo=pypi)](https://pypi.org/project/autogen-ext/) | [![NuGet Contracts](https://img.shields.io/badge/NuGet-Contracts-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Contracts/) <br> [![NuGet Core](https://img.shields.io/badge/NuGet-Core-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core/) <br> [![NuGet Core.Grpc](https://img.shields.io/badge/NuGet-Core.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core.Grpc/) <br> [![NuGet RuntimeGateway.Grpc](https://img.shields.io/badge/NuGet-RuntimeGateway.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.RuntimeGateway.Grpc/) | [![PyPi autogenstudio](https://img.shields.io/badge/PyPi-autogenstudio-purple?logo=pypi)](https://pypi.org/project/autogenstudio/)                          |

</div>

Contribute to the project by following the [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines.

For questions, see our [FAQ](./FAQ.md), [GitHub Discussions](https://github.com/microsoft/autogen/discussions), [Discord server](https://aka.ms/autogen-discord), or [blog](https://devblogs.microsoft.com/autogen/).

## Legal Notices

Microsoft and any contributors grant you a license to the Microsoft documentation and other content
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure, and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at <http://go.microsoft.com/fwlink/?LinkID=254653>.

Privacy information can be found at <https://go.microsoft.com/fwlink/?LinkId=521839>

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel, or otherwise.

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
```
Key improvements and SEO optimizations:

*   **Clear, Concise Title:** "AutoGen: Build Powerful Multi-Agent AI Applications" is more descriptive and keyword-rich.
*   **One-Sentence Hook:**  A compelling opening that summarizes the core benefit.
*   **Keyword Optimization:**  Uses relevant keywords like "multi-agent AI," "AI applications," "framework," and "autonomous."
*   **Structured Headings:**  Uses clear and organized headings (Key Features, Installation, Quickstart, Why Use AutoGen?) for readability and SEO.
*   **Bulleted Lists:**  Highlights key features using bullet points for easy scanning and SEO benefit.
*   **Concise Descriptions:**  Keeps descriptions brief and to the point.
*   **Strong Call to Action:** Encourages exploration and contribution.
*   **Internal Linking:** Uses links within the document to guide users and improve SEO.
*   **Removed Redundancy:**  Eliminated unnecessary phrases and repeated information.
*   **Focused on Benefits:**  Emphasizes the advantages of using AutoGen.
*   **HTML Formatting:** Uses HTML formatting (e.g., `<b>`, `<br>`) to make the content user-friendly and SEO-friendly.
*   **Revised Legal Notices:** Reordered the legal notices section to match the original, maintaining the necessary information while enhancing readability.