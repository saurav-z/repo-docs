<div align="center">
<img src="https://microsoft.github.io/autogen/0.2/img/ag.svg" alt="AutoGen Logo" width="100">

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40pyautogen)](https://twitter.com/pyautogen)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Company?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/105812540)
[![Discord](https://img.shields.io/badge/discord-chat-green?logo=discord)](https://aka.ms/autogen-discord)
[![Documentation](https://img.shields.io/badge/Documentation-AutoGen-blue?logo=read-the-docs)](https://microsoft.github.io/autogen/)
[![Blog](https://img.shields.io/badge/Blog-AutoGen-blue?logo=blogger)](https://devblogs.microsoft.com/autogen/)
</div>

# AutoGen: Build Powerful Multi-Agent AI Applications

AutoGen is a versatile framework empowering developers to create cutting-edge multi-agent AI applications that can autonomously solve complex problems or collaborate seamlessly with humans.  Explore the [original repo](https://github.com/microsoft/autogen) for more details.

## Key Features:

*   **Multi-Agent Orchestration:** Easily design and manage collaborative AI workflows with multiple agents.
*   **Extensible Framework:**  Utilize a layered architecture with Core, AgentChat, and Extensions APIs for flexibility and control.
*   **AgentChat API:** Rapidly prototype using a simplified, opinionated API for common multi-agent patterns.
*   **Flexible Deployment:** Supports local and distributed runtimes for diverse application needs.
*   **AutoGen Studio:** Prototype and run multi-agent workflows without coding using a no-code GUI.
*   **Developer Tools:**  Includes a benchmarking suite (AutoGen Bench) to evaluate agent performance.
*   **.NET and Python Support:** Cross-language support for .NET and Python
*   **Community Driven**: Join a thriving ecosystem with weekly office hours and discussions on Discord.

## Installation

AutoGen requires Python 3.10 or later.

```bash
# Install AgentChat and OpenAI client
pip install -U "autogen-agentchat" "autogen-ext[openai]"

# Install AutoGen Studio for no-code GUI
pip install -U "autogenstudio"
```

Check the [releases](https://github.com/microsoft/autogen/releases) for the latest stable versions. For updates from v0.2, refer to the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html).

## Quickstart

### Hello World

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

### MCP Server

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

### Multi-Agent Orchestration

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

Read the [AgentChat documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html) for more advanced multi-agent orchestrations.

### AutoGen Studio

```bash
# Run AutoGen Studio on http://localhost:8080
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Use AutoGen?

<div align="center">
  <img src="autogen-landing.jpg" alt="AutoGen Landing" width="500">
</div>

AutoGen provides a comprehensive ecosystem for creating AI agents, especially for multi-agent workflows through its:

*   **Framework:** Uses a layered and extensible design with Core, AgentChat, and Extensions APIs.
*   **Developer Tools:** Includes AutoGen Studio for no-code multi-agent application development and AutoGen Bench for performance evaluation.
*   **Applications:** Create solutions for various domains, e.g., [Magentic-One](./python/packages/magentic-one-cli/) using AgentChat and Extensions APIs.

Join the vibrant AutoGen community through weekly office hours, the [Discord server](https://aka.ms/autogen-discord), GitHub Discussions, and the [blog](https://devblogs.microsoft.com/autogen/).

## Where to go next?

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

Contribute to AutoGen by following the guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md).  We welcome bug fixes, new features, and documentation improvements.

## Support

Find answers to common questions in our [FAQ](./FAQ.md), and get real-time support on [Discord](https://aka.ms/autogen-discord) and [GitHub Discussions](https://github.com/microsoft/autogen/discussions). Stay updated via our [blog](https://devblogs.microsoft.com/autogen/).

## Legal Notices

See the [LICENSE](LICENSE) and [LICENSE-CODE](LICENSE-CODE) files for licensing information. Microsoft trademarks are listed at <http://go.microsoft.com/fwlink/?LinkID=254653>. Privacy details are available at <https://go.microsoft.com/fwlink/?LinkId=521839>.

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
```
Key improvements and explanations:

*   **SEO Optimization:**  The title and headings now incorporate relevant keywords ("AutoGen," "multi-agent AI," "framework").  The summary is also improved.
*   **One-Sentence Hook:**  The initial sentence immediately grabs attention and highlights the core value proposition.
*   **Clear Structure:**  Uses headings (Key Features, Installation, Quickstart, Why Use AutoGen?, Where to go next?, Contributing, Support, Legal Notices) to organize content and improve readability.
*   **Bulleted Key Features:** Makes it easy for users to quickly scan and understand AutoGen's capabilities.
*   **Action-Oriented Language:** Uses phrases like "Build," "Empowering," and "Easily design" to engage the reader.
*   **Concise Quickstart Sections:**  Keeps the Quickstart sections short, to the point, and easy to follow.
*   **Emphasis on Benefits:** The "Why Use AutoGen?" section focuses on the advantages and the AutoGen ecosystem.
*   **Community Emphasis:** Highlights the thriving community aspect (office hours, Discord, blog).
*   **Call to Action:** Encourages contribution and provides clear guidance on getting support.
*   **Back to Top Links** Adds "back to top" links to enhance navigation.
*   **Links:** The use of links remains similar, but with a greater focus on the repo and other external resources.
*   **Removed redundant information:** Eliminated repetitions of information.

This improved README is more informative, user-friendly, and optimized for search engines.