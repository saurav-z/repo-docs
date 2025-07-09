<div align="center">
<img src="https://microsoft.github.io/autogen/0.2/img/ag.svg" alt="AutoGen Logo" width="100">

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40pyautogen)](https://twitter.com/pyautogen)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Company?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/105812540)
[![Discord](https://img.shields.io/badge/discord-chat-green?logo=discord)](https://aka.ms/autogen-discord)
[![Documentation](https://img.shields.io/badge/Documentation-AutoGen-blue?logo=read-the-docs)](https://microsoft.github.io/autogen/)
[![Blog](https://img.shields.io/badge/Blog-AutoGen-blue?logo=blogger)](https://devblogs.microsoft.com/autogen/)
</div>

<div align="center" style="background-color: rgba(255, 235, 59, 0.5); padding: 10px; border-radius: 5px; margin: 20px 0;">
  <strong>Important:</strong> This is the official project. We are not affiliated with any fork or startup. See our <a href="https://x.com/pyautogen/status/1857264760951296210">statement</a>.
</div>

# AutoGen: Build Multi-Agent AI Applications with Ease

AutoGen is a powerful framework, developed by Microsoft, for building sophisticated multi-agent AI applications that can work autonomously or collaboratively with humans.  [Explore the AutoGen GitHub repository](https://github.com/microsoft/autogen).

## Key Features

*   **Multi-Agent Workflows:** Easily orchestrate complex AI tasks by coordinating multiple agents.
*   **Flexible Architecture:** Built with a layered design for high-level APIs to low-level components.
*   **Extensible:** Extend functionality with custom agents and integrations through core, agentchat, and extensions API.
*   **AutoGen Studio:** Utilize a no-code GUI to design and run multi-agent workflows.
*   **Community and Support:** Benefit from a thriving ecosystem with weekly office hours, a Discord server, and GitHub Discussions.

## Getting Started

### Installation

AutoGen requires **Python 3.10 or later**.

```bash
# Install AgentChat and OpenAI client from Extensions
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

*   **Important:**  If upgrading from AutoGen v0.2, consult the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html).

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

Create a group chat team for web browsing:

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

Launch AutoGen Studio to prototype and run workflows visually:

```bash
# Run AutoGen Studio on http://localhost:8080
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Choose AutoGen?

AutoGen empowers you to create AI agents and multi-agent workflows with a modular, extensible framework. The ecosystem includes:

*   **Core API:** Implements message passing, event-driven agents, and local/distributed runtimes.
*   **AgentChat API:** Simplifies rapid prototyping with pre-built multi-agent patterns.
*   **Extensions API:** Adds functionality with custom LLM clients and other extensions.
*   **Developer Tools:**
    *   **AutoGen Studio:** Build multi-agent applications with a no-code GUI.
    *   **AutoGen Bench:** Evaluate agent performance using a benchmarking suite.

The AutoGen framework and tools enable you to create applications for your specific domain.

## Resources

<div align="center">

|               | [![Python](https://img.shields.io/badge/AutoGen-Python-blue?logo=python&logoColor=white)](./python)                                                                                                                                                                                                                                                                                                                | [![.NET](https://img.shields.io/badge/AutoGen-.NET-green?logo=.net&logoColor=white)](./dotnet) | [![Studio](https://img.shields.io/badge/AutoGen-Studio-purple?logo=visual-studio&logoColor=white)](./python/packages/autogen-studio)                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Installation  | [![Installation](https://img.shields.io/badge/Install-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html)                                                                                                                                                                                                                                                            | [![Install](https://img.shields.io/badge/Install-green)](https://microsoft.github.io/autogen/dotnet/dev/core/installation.html) | [![Install](https://img.shields.io/badge/Install-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/installation.html) |
| Quickstart    | [![Quickstart](https://img.shields.io/badge/Quickstart-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html#)                                                                                                                                                                                                                                                            | [![Quickstart](https://img.shields.io/badge/Quickstart-green)](https://microsoft.github.io/autogen/dotnet/dev/core/index.html) | [![Usage](https://img.shields.io/badge/Quickstart-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| Tutorial      | [![Tutorial](https://img.shields.io/badge/Tutorial-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/index.html)                                                                                                                                                                                                                                                            | [![Tutorial](https://img.shields.io/badge/Tutorial-green)](https://microsoft.github.io/autogen/dotnet/dev/core/tutorial.html) | [![Usage](https://img.shields.io/badge/Tutorial-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| API Reference | [![API](https://img.shields.io/badge/Docs-blue)](https://microsoft.github.io/autogen/stable/reference/index.html#)                                                                                                                                                                                                                                                                                                    | [![API](https://img.shields.io/badge/Docs-green)](https://microsoft.github.io/autogen/dotnet/dev/api/Microsoft.AutoGen.Contracts.html) | [![API](https://img.shields.io/badge/Docs-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html)               |
| Packages      | [![PyPi autogen-core](https://img.shields.io/badge/PyPi-autogen--core-blue?logo=pypi)](https://pypi.org/project/autogen-core/) <br> [![PyPi autogen-agentchat](https://img.shields.io/badge/PyPi-autogen--agentchat-blue?logo=pypi)](https://pypi.org/project/autogen-agentchat/) <br> [![PyPi autogen-ext](https://img.shields.io/badge/PyPi-autogen--ext-blue?logo=pypi)](https://pypi.org/project/autogen-ext/) | [![NuGet Contracts](https://img.shields.io/badge/NuGet-Contracts-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Contracts/) <br> [![NuGet Core](https://img.shields.io/badge/NuGet-Core-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core/) <br> [![NuGet Core.Grpc](https://img.shields.io/badge/NuGet-Core.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core.Grpc/) <br> [![NuGet RuntimeGateway.Grpc](https://img.shields.io/badge/NuGet-RuntimeGateway.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.RuntimeGateway.Grpc/) | [![PyPi autogenstudio](https://img.shields.io/badge/PyPi-autogenstudio-purple?logo=pypi)](https://pypi.org/project/autogenstudio/)                       |

</div>

## Contribute

We welcome contributions!  See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

*   **Questions?**  Check the [FAQ](./FAQ.md), or ask in [GitHub Discussions](https://github.com/microsoft/autogen/discussions) or our [Discord server](https://aka.ms/autogen-discord).
*   **Stay Updated:** Read the [blog](https://devblogs.microsoft.com/autogen/) for the latest news.

## Legal Notices

(Legal notices content)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
```
Key improvements and SEO considerations:

*   **Clear Title and Hook:**  The title is now clear, and the hook grabs attention.
*   **Targeted Keywords:**  "AutoGen," "multi-agent AI," "AI agents," "framework," and related terms are used naturally throughout the document.
*   **Headings:**  Uses clear headings (e.g., "Key Features," "Getting Started," "Why Choose AutoGen?") to improve readability and SEO.
*   **Bulleted Lists:**  Uses bulleted lists for key features and other important information, making it easy to scan.
*   **Concise Language:**  Phrases are more active and to the point.
*   **Internal Linking:** Links within the README guide users through different sections.
*   **Call to Action:** Promotes the benefits of using the library.
*   **Removed Redundancy:** Streamlined the original text for better flow.
*   **Optimized Quickstart:** Kept the quickstart code examples.
*   **SEO Optimized Content:** This version of the README is designed to attract users who search for multi-agent AI frameworks.