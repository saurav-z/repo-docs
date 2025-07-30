<div align="center">
<img src="https://microsoft.github.io/autogen/0.2/img/ag.svg" alt="AutoGen Logo" width="100">
</div>

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40pyautogen)](https://twitter.com/pyautogen)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Company?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/105812540)
[![Discord](https://img.shields.io/badge/discord-chat-green?logo=discord)](https://aka.ms/autogen-discord)
[![Documentation](https://img.shields.io/badge/Documentation-AutoGen-blue?logo=read-the-docs)](https://microsoft.github.io/autogen/)
[![Blog](https://img.shields.io/badge/Blog-AutoGen-blue?logo=blogger)](https://devblogs.microsoft.com/autogen/)

<div align="center" style="background-color: rgba(255, 235, 59, 0.5); padding: 10px; border-radius: 5px; margin: 20px 0;">
  <strong>Important:</strong> This is the official project. We are not affiliated with any fork or startup. See our <a href="https://x.com/pyautogen/status/1857264760951296210">statement</a>.
</div>

# AutoGen: Build Powerful Multi-Agent AI Applications

**AutoGen** is a versatile framework empowering you to build, experiment with, and deploy sophisticated multi-agent AI applications that work autonomously or collaborate with humans.  [Explore the AutoGen repository](https://github.com/microsoft/autogen).

## Key Features:

*   **Multi-Agent Workflows:** Design and orchestrate complex AI workflows with multiple agents.
*   **Flexible Architecture:** Leverage a layered and extensible design for various abstraction levels, from high-level APIs to low-level components.
*   **Rapid Prototyping:** Use the AgentChat API for quick and efficient development of multi-agent systems.
*   **Extensible Ecosystem:** Utilize the Extensions API to integrate with LLM clients (like OpenAI, AzureOpenAI) and expand functionality.
*   **No-Code GUI:** Employ AutoGen Studio for visual prototyping and running multi-agent applications without coding.
*   **Benchmarking Tools:** Evaluate agent performance using the AutoGen Bench suite.
*   **.NET Support:** AutoGen also has .NET support.

## Getting Started

### Installation

AutoGen requires **Python 3.10 or later**.

```bash
# Install AgentChat and OpenAI client from Extensions
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

For upgrading from AutoGen v0.2, please refer to the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html).

```bash
# Install AutoGen Studio for no-code GUI
pip install -U "autogenstudio"
```

### Quickstart

#### Hello World

Create an assistant agent using OpenAI's GPT-4o model:

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

#### Web Browsing Agent Team

Create a web surfing agent and a user proxy agent:

```python
# pip install -U autogen-agentchat autogen-ext[openai,web-surfer]
# playwright install
import asyncio
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    # The web surfer will open a Chromium browser window to perform web browsing tasks.
    web_surfer = MultimodalWebSurfer("web_surfer", model_client, headless=False, animate_actions=True)
    # The user proxy agent is used to get user input after each step of the web surfer.
    # NOTE: you can skip input by pressing Enter.
    user_proxy = UserProxyAgent("user_proxy")
    # The termination condition is set to end the conversation when the user types 'exit'.
    termination = TextMentionTermination("exit", sources=["user_proxy"])
    # Web surfer and user proxy take turns in a round-robin fashion.
    team = RoundRobinGroupChat([web_surfer, user_proxy], termination_condition=termination)
    try:
        # Start the team and wait for it to terminate.
        await Console(team.run_stream(task="Find information about AutoGen and write a short summary."))
    finally:
        await web_surfer.close()
        await model_client.close()

asyncio.run(main())
```

#### AutoGen Studio

Use AutoGen Studio to prototype and run multi-agent workflows without writing code:

```bash
# Run AutoGen Studio on http://localhost:8080
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Choose AutoGen?

AutoGen provides a complete ecosystem for building and deploying AI agents, including a framework, developer tools, and applications.

*   **Framework:** The framework features a layered design:
    *   **Core API:** Core API implements message passing, event-driven agents, and local and distributed runtime for flexibility and power. It also supports cross-language support for .NET and Python.
    *   **AgentChat API:** A higher-level API for rapid prototyping.
    *   **Extensions API:** Enables integration with LLM clients and expanding capabilities, like code execution.
*   **Developer Tools:**
    *   **AutoGen Studio:** A no-code GUI for building multi-agent applications.
    *   **AutoGen Bench:** A benchmarking suite for evaluating agent performance.

## Where to Go Next?

Explore these resources:

<div align="center">

|               | [![Python](https://img.shields.io/badge/AutoGen-Python-blue?logo=python&logoColor=white)](./python)                                                                                                                                                                                                                                                                                                                | [![.NET](https://img.shields.io/badge/AutoGen-.NET-green?logo=.net&logoColor=white)](./dotnet) | [![Studio](https://img.shields.io/badge/AutoGen-Studio-purple?logo=visual-studio&logoColor=white)](./python/packages/autogen-studio)                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Installation  | [![Installation](https://img.shields.io/badge/Install-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html)                                                                                                                                                                                                                                                            | [![Install](https://img.shields.io/badge/Install-green)](https://microsoft.github.io/autogen/dotnet/dev/core/installation.html) | [![Install](https://img.shields.io/badge/Install-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/installation.html) |
| Quickstart    | [![Quickstart](https://img.shields.io/badge/Quickstart-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html#)                                                                                                                                                                                                                                                            | [![Quickstart](https://img.shields.io/badge/Quickstart-green)](https://microsoft.github.io/autogen/dotnet/dev/core/index.html) | [![Usage](https://img.shields.io/badge/Quickstart-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| Tutorial      | [![Tutorial](https://img.shields.io/badge/Tutorial-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/index.html)                                                                                                                                                                                                                                                            | [![Tutorial](https://img.shields.io/badge/Tutorial-green)](https://microsoft.github.io/autogen/dotnet/dev/core/tutorial.html) | [![Usage](https://img.shields.io/badge/Tutorial-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| API Reference | [![API](https://img.shields.io/badge/Docs-blue)](https://microsoft.github.io/autogen/stable/reference/index.html#)                                                                                                                                                                                                                                                                                                    | [![API](https://img.shields.io/badge/Docs-green)](https://microsoft.github.io/autogen/dotnet/dev/api/Microsoft.AutoGen.Contracts.html) | [![API](https://img.shields.io/badge/Docs-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html)               |
| Packages      | [![PyPi autogen-core](https://img.shields.io/badge/PyPi-autogen--core-blue?logo=pypi)](https://pypi.org/project/autogen-core/) <br> [![PyPi autogen-agentchat](https://img.shields.io/badge/PyPi-autogen--agentchat-blue?logo=pypi)](https://pypi.org/project/autogen-agentchat/) <br> [![PyPi autogen-ext](https://img.shields.io/badge/PyPi-autogen--ext-blue?logo=pypi)](https://pypi.org/project/autogen-ext/) | [![NuGet Contracts](https://img.shields.io/badge/NuGet-Contracts-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Contracts/) <br> [![NuGet Core](https://img.shields.io/badge/NuGet-Core-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core/) <br> [![NuGet Core.Grpc](https://img.shields.io/badge/NuGet-Core.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core.Grpc/) <br> [![NuGet RuntimeGateway.Grpc](https://img.shields.io/badge/NuGet-RuntimeGateway.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.RuntimeGateway.Grpc/) | [![PyPi autogenstudio](https://img.shields.io/badge/PyPi-autogenstudio-purple?logo=pypi)](https://pypi.org/project/autogenstudio/)                       |

</div>

## Contributing

Contribute to AutoGen and become part of our thriving community.  See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Get Support

Have questions? Check out our [FAQ](./FAQ.md).  Engage with the community on [GitHub Discussions](https://github.com/microsoft/autogen/discussions) or [Discord](https://aka.ms/autogen-discord).  Stay updated via our [blog](https://devblogs.microsoft.com/autogen/).

## Legal Notices

(Same as original)