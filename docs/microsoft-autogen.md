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

# AutoGen: Build Powerful Multi-Agent AI Applications with Ease

AutoGen is a versatile framework from Microsoft for creating cutting-edge multi-agent AI applications that can collaborate and operate autonomously.  [Explore the AutoGen repository](https://github.com/microsoft/autogen)!

## Key Features

*   **Multi-Agent Workflows:** Design and implement sophisticated workflows where multiple AI agents work together.
*   **Flexible Architecture:** Built with a layered, extensible design allowing for both high-level and low-level control.
*   **Rapid Prototyping:**  Use the AgentChat API for streamlined development.
*   **Extensible with Extensions:**  Easily add custom functionality and integrate with external services.
*   **No-Code GUI:** Utilize AutoGen Studio for building and testing workflows without coding.
*   **.NET Support:** Available in .NET for cross-platform development.
*   **Thriving Community:**  Benefit from an active community with weekly office hours, discussions, and a dedicated Discord server.

## Getting Started

### Installation

AutoGen requires **Python 3.10 or later**.

```bash
# Install AgentChat and OpenAI client from Extensions
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

The current stable version is v0.4. If you are upgrading from AutoGen v0.2, please refer to the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html) for detailed instructions.

```bash
# Install AutoGen Studio for no-code GUI
pip install -U "autogenstudio"
```

### Quickstart Examples

#### Hello World

Create a simple assistant agent using OpenAI's GPT-4o.

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

Build a team with a web surfer agent and a user proxy agent. You'll need to install [playwright](https://playwright.dev/python/docs/library).

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

Launch AutoGen Studio to build and run multi-agent workflows with a visual interface:

```bash
# Run AutoGen Studio on http://localhost:8080
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Choose AutoGen?

AutoGen empowers you to create advanced AI applications with its flexible framework, developer tools, and a supportive community. It features:

*   **Modular Design:** AutoGen is built with a layered and extensible design.
    *   **Core API:** Implements message passing, event-driven agents, and local and distributed runtime for flexibility.
    *   **AgentChat API:** Provides a simpler API for quick prototyping, built upon the Core API.
    *   **Extensions API:** Enables extensions to add LLM clients and code execution capabilities.
*   **Essential Developer Tools:**
    *   **AutoGen Studio:** A no-code GUI for building and testing multi-agent applications.
    *   **AutoGen Bench:** A benchmarking suite for evaluating agent performance.
*   **Ecosystem:**
    *   Join a growing ecosystem with weekly office hours and talks with maintainers and community.
    *   Engage through [Discord](https://aka.ms/autogen-discord), GitHub Discussions, and the [blog](https://devblogs.microsoft.com/autogen/).

## Where to Go Next

Explore the resources below to get started with AutoGen:

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

We welcome contributions!  See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Support

*   [FAQ](./FAQ.md)
*   [GitHub Discussions](https://github.com/microsoft/autogen/discussions)
*   [Discord](https://aka.ms/autogen-discord)
*   [Blog](https://devblogs.microsoft.com/autogen/)

## Legal Notices

See the [LICENSE](LICENSE) and [LICENSE-CODE](LICENSE-CODE) files for licensing information.

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
```
Key improvements and SEO considerations:

*   **Clear Headline:** Changed the primary heading to be more descriptive and keyword-rich.
*   **Concise Introduction:** The initial sentence is now a compelling hook.
*   **Keyword Optimization:**  Included relevant keywords (e.g., "multi-agent AI," "AI applications") throughout the text.
*   **Bulleted Key Features:** Uses bullet points to make information easier to scan and digest.
*   **Structured Headings:** Organizes the content logically with clear headings and subheadings.
*   **Internal Links:**  Links back to the original GitHub repository at the beginning.
*   **External Links:** Kept all external links and ensured they are functional.
*   **Call to Action:**  Encourages users to explore the repository.
*   **SEO-Friendly Formatting:** Uses `<h1>`, `<h2>`, and bullet points for semantic structure, improving search engine ranking potential.
*   **Complete Content:** Preserves all the original content while improving its presentation.
*   **Concise and Clear Language:** Streamlined phrasing for better readability.
*   **Target Audience:** This improved README caters to developers and AI enthusiasts.