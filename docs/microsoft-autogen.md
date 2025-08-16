<div align="center">
<img src="https://microsoft.github.io/autogen/0.2/img/ag.svg" alt="AutoGen Logo" width="100">

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40pyautogen)](https://twitter.com/pyautogen)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Company?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/105812540)
[![Discord](https://img.shields.io/badge/discord-chat-green?logo=discord)](https://aka.ms/autogen-discord)
[![Documentation](https://img.shields.io/badge/Documentation-AutoGen-blue?logo=read-the-docs)](https://microsoft.github.io/autogen/)
[![Blog](https://img.shields.io/badge/Blog-AutoGen-blue?logo=blogger)](https://devblogs.microsoft.com/autogen/)
</div>

# AutoGen: Build Powerful Multi-Agent AI Applications

**AutoGen** is a versatile framework enabling developers to create sophisticated multi-agent AI applications that can operate autonomously or collaboratively with humans. ([View the original repository](https://github.com/microsoft/autogen))

## Key Features

*   **Multi-Agent Workflows:** Design and implement complex AI workflows with multiple agents.
*   **Extensible Architecture:** Leverage a layered design with core APIs, agent chat APIs, and extension APIs for flexibility and customization.
*   **Developer Tools:** Utilize AutoGen Studio (no-code GUI) and AutoGen Bench for efficient development and evaluation.
*   **Wide Range of Applications:** Create solutions for various domains with support for web browsing, code execution, file handling, and more.
*   **.NET Support:** AutoGen also offers .NET support for cross-platform development.

## Installation

Ensure you have Python 3.10 or later installed.

```bash
# Install AgentChat and OpenAI client from Extensions
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

For AutoGen Studio:

```bash
# Install AutoGen Studio for no-code GUI
pip install -U "autogenstudio"
```

**Important:** If upgrading from AutoGen v0.2, consult the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html) for detailed instructions.

## Quickstart Examples

### Hello World

Create a simple assistant agent using OpenAI's GPT-4o model:

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

> **Warning**: Only connect to trusted MCP servers as they may execute commands
> in your local environment or expose sensitive information.

### AutoGen Studio

Rapidly prototype and run multi-agent workflows with the no-code GUI:

```bash
# Run AutoGen Studio on http://localhost:8080
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Choose AutoGen?

<div align="center">
  <img src="autogen-landing.jpg" alt="AutoGen Landing" width="500">
</div>

AutoGen provides a comprehensive ecosystem for building AI agents, particularly multi-agent systems, including a flexible framework, powerful developer tools, and ready-to-use applications.

*   **Framework:**
    *   **Core API:**  Provides message passing, event-driven agents, and flexible runtime options.
    *   **AgentChat API:**  Simplifies prototyping with pre-built multi-agent patterns.
    *   **Extensions API:**  Offers extendability through the support for LLM clients and capabilities like code execution.

*   **Developer Tools:**
    *   **AutoGen Studio:** A no-code GUI for visually building and testing multi-agent applications.
    *   **AutoGen Bench:**  A benchmarking suite for evaluating the performance of your AI agents.

*   **Ecosystem:** A thriving community with weekly office hours, discussions, a Discord server for real-time support, and a blog with tutorials and updates.

## Where to Go Next?

Explore the following resources to get started:

<div align="center">

|               | [![Python](https://img.shields.io/badge/AutoGen-Python-blue?logo=python&logoColor=white)](./python)                                                                                                                                                                                                                                                                                                                | [![.NET](https://img.shields.io/badge/AutoGen-.NET-green?logo=.net&logoColor=white)](./dotnet)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | [![Studio](https://img.shields.io/badge/AutoGen-Studio-purple?logo=visual-studio&logoColor=white)](./python/packages/autogen-studio)                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Installation  | [![Installation](https://img.shields.io/badge/Install-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html)                                                                                                                                                                                                                                                         | [![Install](https://img.shields.io/badge/Install-green)](https://microsoft.github.io/autogen/dotnet/dev/core/installation.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | [![Install](https://img.shields.io/badge/Install-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/installation.html) |
| Quickstart    | [![Quickstart](https://img.shields.io/badge/Quickstart-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html#)                                                                                                                                                                                                                                                         | [![Quickstart](https://img.shields.io/badge/Quickstart-green)](https://microsoft.github.io/autogen/dotnet/dev/core/index.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | [![Usage](https://img.shields.io/badge/Quickstart-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)      |
| Tutorial      | [![Tutorial](https://img.shields.io/badge/Tutorial-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/index.html)                                                                                                                                                                                                                                                          | [![Tutorial](https://img.shields.io/badge/Tutorial-green)](https://microsoft.github.io/autogen/dotnet/dev/core/tutorial.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | [![Usage](https://img.shields.io/badge/Tutorial-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| API Reference | [![API](https://img.shields.io/badge/Docs-blue)](https://microsoft.github.io/autogen/stable/reference/index.html#)                                                                                                                                                                                                                                                                                                 | [![API](https://img.shields.io/badge/Docs-green)](https://microsoft.github.io/autogen/dotnet/dev/api/Microsoft.AutoGen.Contracts.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | [![API](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html)               |
| Packages      | [![PyPi autogen-core](https://img.shields.io/badge/PyPi-autogen--core-blue?logo=pypi)](https://pypi.org/project/autogen-core/) <br> [![PyPi autogen-agentchat](https://img.shields.io/badge/PyPi-autogen--agentchat-blue?logo=pypi)](https://pypi.org/project/autogen-agentchat/) <br> [![PyPi autogen-ext](https://img.shields.io/badge/PyPi-autogen--ext-blue?logo=pypi)](https://pypi.org/project/autogen-ext/) | [![NuGet Contracts](https://img.shields.io/badge/NuGet-Contracts-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Contracts/) <br> [![NuGet Core](https://img.shields.io/badge/NuGet-Core-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core/) <br> [![NuGet Core.Grpc](https://img.shields.io/badge/NuGet-Core.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core.Grpc/) <br> [![NuGet RuntimeGateway.Grpc](https://img.shields.io/badge/NuGet-RuntimeGateway.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.RuntimeGateway.Grpc/) | [![PyPi autogenstudio](https://img.shields.io/badge/PyPi-autogenstudio-purple?logo=pypi)](https://pypi.org/project/autogenstudio/)                          |

</div>

## Contributing

We welcome contributions! Review [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Support

*   **FAQ:** [FAQ](./FAQ.md)
*   **Discussions:** [GitHub Discussions](https://github.com/microsoft/autogen/discussions)
*   **Discord:** [Discord server](https://aka.ms/autogen-discord)
*   **Blog:** [Blog](https://devblogs.microsoft.com/autogen/)

## Legal Notices

(Same as original)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
```
Key improvements and SEO considerations:

*   **Clear Title:** Uses the most relevant keyword ("AutoGen") and a strong value proposition ("Build Powerful Multi-Agent AI Applications").
*   **One-Sentence Hook:**  Provides a concise and engaging introduction.
*   **Keyword Optimization:**  Includes relevant keywords throughout the headings and text (e.g., "multi-agent AI," "AI agents," "framework," "developer tools").
*   **Structured Headings:** Uses clear, descriptive headings for better readability and SEO (Installation, Quickstart, Why Use AutoGen?, Where to go next? etc.).
*   **Bulleted Key Features:** Highlights the key benefits in a scannable format, making it easy for users to understand the value proposition.
*   **Internal Links:** Keeps the links to documentation and other pages inside the table.
*   **Call to Action:** Encourages contribution and participation.
*   **Concise and Informative:**  Provides essential information without being overly verbose.
*   **Clear Instructions:**  Includes install commands in the right places.
*   **Markdown formatting:**  Uses Markdown for readability.
*   **Link Back:** Includes the original repository link in the first sentence.