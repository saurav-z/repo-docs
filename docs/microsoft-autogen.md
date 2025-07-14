<div align="center">
<img src="https://microsoft.github.io/autogen/0.2/img/ag.svg" alt="AutoGen Logo" width="100">
</div>

<div align="center">
    <a href="https://twitter.com/pyautogen"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40pyautogen" alt="Twitter"/></a>
    <a href="https://www.linkedin.com/company/105812540"><img src="https://img.shields.io/badge/LinkedIn-Company?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
    <a href="https://aka.ms/autogen-discord"><img src="https://img.shields.io/badge/discord-chat-green?logo=discord" alt="Discord"/></a>
    <a href="https://microsoft.github.io/autogen/"><img src="https://img.shields.io/badge/Documentation-AutoGen-blue?logo=read-the-docs" alt="Documentation"/></a>
    <a href="https://devblogs.microsoft.com/autogen/"><img src="https://img.shields.io/badge/Blog-AutoGen-blue?logo=blogger" alt="Blog"/></a>
</div>

<div align="center" style="background-color: rgba(255, 235, 59, 0.5); padding: 10px; border-radius: 5px; margin: 20px 0;">
  <strong>Important:</strong> This is the official project. We are not affiliated with any fork or startup. See our <a href="https://x.com/pyautogen/status/1857264760951296210">statement</a>.
</div>

# AutoGen: Build Powerful Multi-Agent AI Applications

**AutoGen is a versatile framework enabling developers to create sophisticated multi-agent AI applications that can work autonomously or in collaboration with humans, making complex tasks more manageable and efficient.**  [Check out the original repo](https://github.com/microsoft/autogen) for further details.

## Key Features:

*   **Multi-Agent Framework:** Design and implement multi-agent workflows.
*   **Autonomous or Collaborative Agents:** Agents can operate independently or interact with human users.
*   **Extensible Design:** Layered architecture allows for flexible use, from high-level APIs to low-level components.
*   **Developer Tools:**
    *   AutoGen Studio: A no-code GUI for building and testing multi-agent applications.
    *   AutoGen Bench: A suite for evaluating agent performance.
*   **.NET Support:**  Offers cross-language support with .NET.
*   **Community Support:** Active community with office hours, Discord server, and GitHub Discussions.

## Installation

AutoGen requires **Python 3.10 or later**.

```bash
# Install AgentChat and OpenAI client from Extensions
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

The current stable version is v0.4. If you are upgrading from AutoGen v0.2, please refer to the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html) for detailed instructions on how to update your code and configurations.

```bash
# Install AutoGen Studio for no-code GUI
pip install -U "autogenstudio"
```

## Quickstart Examples

### Hello World

Create an assistant agent using OpenAI's GPT-4o model. See [other supported models](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/models.html).

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

### Web Browsing Agent Team

Create a group chat team with a web surfer agent and a user proxy agent
for web browsing tasks. You need to install [playwright](https://playwright.dev/python/docs/library).

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

### AutoGen Studio

Use AutoGen Studio to prototype and run multi-agent workflows without writing code.

```bash
# Run AutoGen Studio on http://localhost:8080
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Use AutoGen?

<div align="center">
  <img src="autogen-landing.jpg" alt="AutoGen Landing" width="500">
</div>

AutoGen provides a comprehensive ecosystem for creating and managing AI agents, particularly for multi-agent workflows. This ecosystem offers:

*   **Framework**:
    *   **Core API:** Implements message passing, event-driven agents, and local/distributed runtimes for flexibility.  Supports cross-language use (.NET and Python).
    *   **AgentChat API:**  An opinionated API for rapid prototyping, built on top of Core API.
    *   **Extensions API:** Allows for expanding capabilities, including LLM clients (OpenAI, AzureOpenAI) and code execution.
*   **Developer Tools:**
    *   **AutoGen Studio:** A no-code GUI for creating multi-agent applications.
    *   **AutoGen Bench:**  A benchmarking suite for evaluating agent performance.

## Where to go next?

<div align="center">

|               | [![Python](https://img.shields.io/badge/AutoGen-Python-blue?logo=python&logoColor=white)](./python)                                                                                                                                                                                                                                                                                                                | [![.NET](https://img.shields.io/badge/AutoGen-.NET-green?logo=.net&logoColor=white)](./dotnet) | [![Studio](https://img.shields.io/badge/AutoGen-Studio-purple?logo=visual-studio&logoColor=white)](./python/packages/autogen-studio)                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Installation  | [![Installation](https://img.shields.io/badge/Install-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html)                                                                                                                                                                                                                                                            | [![Install](https://img.shields.io/badge/Install-green)](https://microsoft.github.io/autogen/dotnet/dev/core/installation.html) | [![Install](https://img.shields.io/badge/Install-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/installation.html) |
| Quickstart    | [![Quickstart](https://img.shields.io/badge/Quickstart-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html#)                                                                                                                                                                                                                                                            | [![Quickstart](https://img.shields.io/badge/Quickstart-green)](https://microsoft.github.io/autogen/dotnet/dev/core/index.html) | [![Usage](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| Tutorial      | [![Tutorial](https://img.shields.io/badge/Tutorial-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/index.html)                                                                                                                                                                                                                                                            | [![Tutorial](https://microsoft.github.io/autogen/dotnet/dev/core/tutorial.html) | [![Usage](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| API Reference | [![API](https://img.shields.io/badge/Docs-blue)](https://microsoft.github.io/autogen/stable/reference/index.html#)                                                                                                                                                                                                                                                                                                    | [![API](https://microsoft.github.io/autogen/dotnet/dev/api/Microsoft.AutoGen.Contracts.html) | [![API](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html)               |
| Packages      | [![PyPi autogen-core](https://img.shields.io/badge/PyPi-autogen--core-blue?logo=pypi)](https://pypi.org/project/autogen-core/) <br> [![PyPi autogen-agentchat](https://img.shields.io/badge/PyPi-autogen--agentchat-blue?logo=pypi)](https://pypi.org/project/autogen-agentchat/) <br> [![PyPi autogen-ext](https://img.shields.io/badge/PyPi-autogen--ext-blue?logo=pypi)](https://pypi.org/project/autogen-ext/) | [![NuGet Contracts](https://img.shields.io/badge/NuGet-Contracts-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Contracts/) <br> [![NuGet Core](https://img.shields.io/badge/NuGet-Core-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core/) <br> [![NuGet Core.Grpc](https://img.shields.io/badge/NuGet-Core.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core.Grpc/) <br> [![NuGet RuntimeGateway.Grpc](https://img.shields.io/badge/NuGet-RuntimeGateway.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.RuntimeGateway.Grpc/) | [![PyPi autogenstudio](https://img.shields.io/badge/PyPi-autogenstudio-purple?logo=pypi)](https://pypi.org/project/autogenstudio/)                       |

</div>

## Contributing

Interested in contributing? See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines. We welcome contributions of all kinds.

## Support

Have questions?  Consult our [FAQ](./FAQ.md), [GitHub Discussions](https://github.com/microsoft/autogen/discussions), or [Discord server](https://aka.ms/autogen-discord). You can also read our [blog](https://devblogs.microsoft.com/autogen/) for updates.

## Legal Notices

(Content from original README)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:** Starts with a strong sentence highlighting the core value proposition.
*   **Keyword Optimization:**  Includes relevant keywords like "multi-agent AI," "AI applications," "framework," and specific tool names.
*   **Structured Headings:** Uses clear H2 and H3 headings for improved readability and SEO.
*   **Bulleted Key Features:** Makes the core benefits easy to scan and understand.
*   **Internal Linking:**  Includes links to key sections and documentation.
*   **External Links:** Links to the official project, community resources, and relevant external pages.
*   **Concise Language:** Removes unnecessary words and phrases.
*   **Focus on Value:** Emphasizes the benefits of using AutoGen.
*   **Updated Badges:** Includes relevant badges and links.
*   **Back to Top:** Preserves the "Back to Top" link for easier navigation.