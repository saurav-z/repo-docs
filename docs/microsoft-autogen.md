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

# AutoGen: Build Powerful Multi-Agent AI Applications

AutoGen is a versatile framework from Microsoft for building and deploying cutting-edge multi-agent AI applications that can operate autonomously or collaboratively with humans.  [Explore the AutoGen repository on GitHub](https://github.com/microsoft/autogen).

## Key Features

*   **Multi-Agent Workflows:** Design and implement complex workflows involving multiple AI agents.
*   **Extensible Architecture:** Built on a layered design for flexibility, allowing you to leverage high-level APIs or customize low-level components.
*   **Rapid Prototyping:** Quickly build and test multi-agent systems with the AgentChat API.
*   **No-Code GUI:** Utilize AutoGen Studio for a visual interface to prototype, experiment, and run agent workflows.
*   **Extensive Ecosystem:**  Access a range of developer tools, pre-built extensions, and a thriving community for support and collaboration.
*   **.NET and Python Support**: Built for cross-language support.
*   **Web Browsing and Code Execution:** Built in capabilities that allow agents to perform web browsing and code execution.

## Getting Started

### Installation

AutoGen requires **Python 3.10 or later**.

```bash
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

The current stable version is v0.4. If you are upgrading from AutoGen v0.2, please refer to the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html) for detailed instructions on how to update your code and configurations.

```bash
pip install -U "autogenstudio"
```

### Quickstart Examples

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

Build a team with a web surfer agent and a user proxy agent:

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

Launch AutoGen Studio for a no-code interface:

```bash
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Use AutoGen?

<div align="center">
  <img src="autogen-landing.jpg" alt="AutoGen Landing" width="500">
</div>

AutoGen offers a comprehensive ecosystem to create powerful AI agents and multi-agent workflows. Key components include:

*   **Framework:** A layered design with Core, AgentChat, and Extensions APIs, offering flexibility from high-level APIs to low-level components.
    *   **Core API:** Provides message passing, event-driven agents, and local/distributed runtime. Supports cross-language support for .NET and Python.
    *   **AgentChat API:** Enables rapid prototyping with a simpler, opinionated API built on top of the Core API.
    *   **Extensions API:** Integrates LLM clients (e.g., OpenAI, AzureOpenAI) and capabilities like code execution.
*   **Developer Tools:**
    *   [AutoGen Studio](./python/packages/autogen-studio/): A no-code GUI for building multi-agent applications.
    *   [AutoGen Bench](./python/packages/agbench/): A benchmarking suite for evaluating agent performance.

AutoGen empowers you to create applications for your domain.

## Next Steps

<div align="center">

|               | [![Python](https://img.shields.io/badge/AutoGen-Python-blue?logo=python&logoColor=white)](./python)                                                                                                                                                                                                                                                                                                                | [![.NET](https://img.shields.io/badge/AutoGen-.NET-green?logo=.net&logoColor=white)](./dotnet) | [![Studio](https://img.shields.io/badge/AutoGen-Studio-purple?logo=visual-studio&logoColor=white)](./python/packages/autogen-studio)                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Installation  | [![Installation](https://img.shields.io/badge/Install-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html)                                                                                                                                                                                                                                                            | [![Install](https://img.shields.io/badge/Install-green)](https://microsoft.github.io/autogen/dotnet/dev/core/installation.html) | [![Install](https://img.shields.io/badge/Install-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/installation.html) |
| Quickstart    | [![Quickstart](https://img.shields.io/badge/Quickstart-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html#)                                                                                                                                                                                                                                                            | [![Quickstart](https://img.shields.io/badge/Quickstart-green)](https://microsoft.github.io/autogen/dotnet/dev/core/index.html) | [![Usage](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| Tutorial      | [![Tutorial](https://img.shields.io/badge/Tutorial-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/index.html)                                                                                                                                                                                                                                                            | [![Tutorial](https://img.shields.io/badge/Tutorial-green)](https://microsoft.github.io/autogen/dotnet/dev/core/tutorial.html) | [![Usage](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| API Reference | [![API](https://img.shields.io/badge/Docs-blue)](https://microsoft.github.io/autogen/stable/reference/index.html#)                                                                                                                                                                                                                                                                                                    | [![API](https://img.shields.io/badge/Docs-green)](https://microsoft.github.io/autogen/dotnet/dev/api/Microsoft.AutoGen.Contracts.html) | [![API](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html)               |
| Packages      | [![PyPi autogen-core](https://img.shields.io/badge/PyPi-autogen--core-blue?logo=pypi)](https://pypi.org/project/autogen-core/) <br> [![PyPi autogen-agentchat](https://img.shields.io/badge/PyPi-autogen--agentchat-blue?logo=pypi)](https://pypi.org/project/autogen-agentchat/) <br> [![PyPi autogen-ext](https://img.shields.io/badge/PyPi-autogen--ext-blue?logo=pypi)](https://pypi.org/project/autogen-ext/) | [![NuGet Contracts](https://img.shields.io/badge/NuGet-Contracts-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Contracts/) <br> [![NuGet Core](https://img.shields.io/badge/NuGet-Core-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core/) <br> [![NuGet Core.Grpc](https://img.shields.io/badge/NuGet-Core.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core.Grpc/) <br> [![NuGet RuntimeGateway.Grpc](https://img.shields.io/badge/NuGet-RuntimeGateway.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.RuntimeGateway.Grpc/) | [![PyPi autogenstudio](https://img.shields.io/badge/PyPi-autogenstudio-purple?logo=pypi)](https://pypi.org/project/autogenstudio/)                       |

</div>

## Contribution and Community

Contribute to AutoGen by reviewing the [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines. Join our community through [GitHub Discussions](https://github.com/microsoft/autogen/discussions) and our [Discord server](https://aka.ms/autogen-discord) for support. Stay informed with our [blog](https://devblogs.microsoft.com/autogen/) for updates.

## Legal Notices

*   [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode)
*   [MIT License](https://opensource.org/licenses/MIT)
*   [Microsoft Trademark Guidelines](http://go.microsoft.com/fwlink/?LinkID=254653)
*   [Privacy Information](https://go.microsoft.com/fwlink/?LinkId=521839)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
```
Key improvements and SEO optimizations:

*   **Clear Title and Hook:**  The title and opening sentence are designed to grab attention and clearly state what AutoGen is.
*   **Keyword Optimization:**  Includes relevant keywords like "multi-agent AI," "AI agents," "framework," "autonomous," and "workflows."
*   **Structured Headings:** Uses clear headings and subheadings for readability and SEO.
*   **Bulleted Key Features:**  Highlights the main selling points of AutoGen in a scannable format.
*   **Concise Language:**  The text is rewritten for brevity and clarity.
*   **Actionable Quickstart:** The Quickstart section remains, but it is more accessible by incorporating the name.
*   **Community and Legal Sections:**  These are kept for completeness, but with some reorganization for better flow.
*   **Internal and External Links:**  Uses internal links where appropriate and provides links to important resources and documentation.
*   **Visual Appeal:**  Maintains the logo and images for engagement.
*   **Clear Call to Action:**  Provides links to relevant resources to encourage engagement (e.g. where to go next and contributing).