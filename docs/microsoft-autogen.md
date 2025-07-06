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

**AutoGen** is a cutting-edge framework from Microsoft for developing and deploying multi-agent AI applications, offering both autonomous operation and collaborative human-AI workflows.

## Key Features

*   **Multi-Agent Framework:** Design and build complex AI workflows with multiple agents.
*   **Flexible Architecture:** Utilize a layered design for varying levels of abstraction, from high-level APIs to low-level components.
*   **Extensible Design:** Leverage the layered architecture to use the framework at different levels of abstraction, from high-level APIs to low-level components.
*   **Extensive API Support:** Includes AgentChat API for rapid prototyping and Core API for message passing.
*   **Developer Tools:**
    *   **AutoGen Studio:** A no-code GUI for building and testing multi-agent applications.
    *   **AutoGen Bench:** A benchmarking suite for evaluating agent performance.
*   **Community and Support:** Benefit from a thriving ecosystem with weekly office hours, Discord, GitHub Discussions, and a blog for tutorials and updates.

## Installation

AutoGen requires Python 3.10 or later.

```bash
# Install AgentChat and OpenAI client from Extensions
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

The current stable version is v0.4.  If upgrading from v0.2, refer to the [Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html).

```bash
# Install AutoGen Studio for no-code GUI
pip install -U "autogenstudio"
```

## Quickstart

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

AutoGen offers a comprehensive ecosystem for building AI agents, including the framework, developer tools, and real-world applications.  This is a great way to create AI agents.

The framework is designed to be layered and extensible.

-   **Core API**: Implements message passing, event-driven agents, and local/distributed runtime for flexibility. Supports cross-language support for .NET and Python.
-   **AgentChat API**: Provides a simpler API for rapid prototyping, built on top of the Core API, and supports multi-agent patterns.
-   **Extensions API**: Enables third-party extensions to expand capabilities, including specific implementations of LLM clients (e.g., OpenAI, AzureOpenAI).

AutoGen also provides essential _developer tools_:

<div align="center">
  <img src="https://media.githubusercontent.com/media/microsoft/autogen/refs/heads/main/python/packages/autogen-studio/docs/ags_screen.png" alt="AutoGen Studio Screenshot" width="500">
</div>

-   [AutoGen Studio](./python/packages/autogen-studio/) provides a no-code GUI for building multi-agent applications.
-   [AutoGen Bench](./python/packages/agbench/) provides a benchmarking suite for evaluating agent performance.

Create applications for your domain. For example, [Magentic-One](./python/packages/magentic-one-cli/) is a state-of-the-art multi-agent team built using AgentChat API and Extensions API that can handle a variety of tasks that require web browsing, code execution, and file handling.

Join the AutoGen community and contribute to a thriving ecosystem! We host weekly office hours and talks with maintainers and community. We also have a [Discord server](https://aka.ms/autogen-discord) for real-time chat, GitHub Discussions for Q&A, and a blog for tutorials and updates.

## Getting Started

|               | [![Python](https://img.shields.io/badge/AutoGen-Python-blue?logo=python&logoColor=white)](./python)                                                                                                                                                                                                                                                                                                                | [![.NET](https://img.shields.io/badge/AutoGen-.NET-green?logo=.net&logoColor=white)](./dotnet) | [![Studio](https://img.shields.io/badge/AutoGen-Studio-purple?logo=visual-studio&logoColor=white)](./python/packages/autogen-studio)                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Installation  | [![Installation](https://img.shields.io/badge/Install-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html)                                                                                                                                                                                                                                                            | [![Install](https://img.shields.io/badge/Install-green)](https://microsoft.github.io/autogen/dotnet/dev/core/installation.html) | [![Install](https://img.shields.io/badge/Install-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/installation.html) |
| Quickstart    | [![Quickstart](https://img.shields.io/badge/Quickstart-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html#)                                                                                                                                                                                                                                                            | [![Quickstart](https://img.shields.io/badge/Quickstart-green)](https://microsoft.github.io/autogen/dotnet/dev/core/index.html) | [![Usage](https://img.shields.io/badge/Quickstart-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| Tutorial      | [![Tutorial](https://img.shields.io/badge/Tutorial-blue)](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/index.html)                                                                                                                                                                                                                                                            | [![Tutorial](https://img.shields.io/badge/Tutorial-green)](https://microsoft.github.io/autogen/dotnet/dev/core/tutorial.html) | [![Usage](https://img.shields.io/badge/Tutorial-purple)](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html#)        |
| API Reference | [![API](https://img.shields.io/badge/Docs-blue)](https://microsoft.github.io/autogen/stable/reference/index.html#)                                                                                                                                                                                                                                                                                                    | [![API](https://img.shields.io/badge/Docs-green)](https://microsoft.github.io/autogen/dotnet/dev/api/Microsoft.AutoGen.Contracts.html) | [![API](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html)               |
| Packages      | [![PyPi autogen-core](https://img.shields.io/badge/PyPi-autogen--core-blue?logo=pypi)](https://pypi.org/project/autogen-core/) <br> [![PyPi autogen-agentchat](https://img.shields.io/badge/PyPi-autogen--agentchat-blue?logo=pypi)](https://pypi.org/project/autogen-agentchat/) <br> [![PyPi autogen-ext](https://img.shields.io/badge/PyPi-autogen--ext-blue?logo=pypi)](https://pypi.org/project/autogen-ext/) | [![NuGet Contracts](https://img.shields.io/badge/NuGet-Contracts-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Contracts/) <br> [![NuGet Core](https://img.shields.io/badge/NuGet-Core-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core/) <br> [![NuGet Core.Grpc](https://img.shields.io/badge/NuGet-Core.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.Core.Grpc/) <br> [![NuGet RuntimeGateway.Grpc](https://img.shields.io/badge/NuGet-RuntimeGateway.Grpc-green?logo=nuget)](https://www.nuget.org/packages/Microsoft.AutoGen.RuntimeGateway.Grpc/) | [![PyPi autogenstudio](https://img.shields.io/badge/PyPi-autogenstudio-purple?logo=pypi)](https://pypi.org/project/autogenstudio/)                       |

## Contribute

Interested in contributing? See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.  Contributions are welcome, including bug fixes, new features, and documentation improvements.

## Support

Have questions? Check out our [FAQ](./FAQ.md) or ask in our [GitHub Discussions](https://github.com/microsoft/autogen/discussions) or join our [Discord server](https://aka.ms/autogen-discord).  You can also read our [blog](https://devblogs.microsoft.com/autogen/) for updates.

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

<p align="center">
    <a href="https://github.com/microsoft/autogen">
        <img src="https://img.shields.io/badge/GitHub-AutoGen-blue?style=for-the-badge&logo=github" alt="GitHub">
    </a>
</p>
```
Key improvements and SEO considerations:

*   **Clear, concise title:** "AutoGen: Build Powerful Multi-Agent AI Applications" includes the project name and a clear benefit.
*   **One-sentence hook:** Placed at the beginning to grab attention.
*   **Keyword-rich headings:** Use of relevant keywords like "Multi-Agent AI," "Framework," "Developer Tools" throughout.
*   **Bulleted key features:** Easy to scan and highlights the main selling points.
*   **Context and explanations:** Provides context and explanations of different features.
*   **Actionable "Getting Started" section:**  Provides direct links to the documentation.
*   **Emphasis on Community:** Mentions of office hours, Discord, GitHub discussions, and blog for community building.
*   **Clear call to action:** Encourages contribution and support.
*   **Proper Markdown formatting:** Makes the README easy to read and visually appealing.
*   **Internal links:** Use of internal links to make the content easier to navigate.
*   **SEO-friendly alt text:** Added alt text to images for accessibility and SEO.
*   **Back to top link:** Keeps the user engaged.
*   **GitHub badge:** Easy to find and directs the user to the source code.
*   **Concise and easy to understand:**  The document has been rewritten to better explain AutoGen for the end user.
*   **Updated Quickstart examples:**  Maintains the original examples, but includes more descriptive comments.
*   **Added Keywords:** Used relevant keywords to improve SEO.