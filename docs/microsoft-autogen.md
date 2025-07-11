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

# AutoGen: Build and Manage Multi-Agent AI Applications with Ease

**AutoGen** is a powerful framework from Microsoft for building and managing multi-agent AI applications, offering flexible and extensible tools for creating autonomous or human-in-the-loop workflows.  [Explore the AutoGen repository](https://github.com/microsoft/autogen) for the latest updates.

## Key Features

*   **Multi-Agent Framework:** Design, implement, and manage complex AI agent interactions.
*   **Extensible Architecture:** Build on a layered design with Core, AgentChat, and Extensions APIs for flexible development.
*   **AutoGen Studio:** A no-code GUI for rapid prototyping and deployment of multi-agent systems.
*   **Benchmarking Tools:** Evaluate and compare agent performance.
*   **Community Support:** Active community with weekly office hours, Discord server, and GitHub Discussions.
*   **Cross-Language Support:**  Includes support for both Python and .NET.

## Getting Started

### Installation

AutoGen requires **Python 3.10 or later**.

```bash
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

*Refer to the [Installation Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html) for detailed instructions.*

### Quickstart: Hello World Example

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

### Example: Web Browsing Agent Team

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

Launch the no-code GUI for developing multi-agent applications:

```bash
autogenstudio ui --port 8080 --appdir ./my-app
```

## Why Choose AutoGen?

AutoGen provides a comprehensive ecosystem for creating AI agents and, especially, multi-agent workflows:

*   **Framework:** A layered design with Core, AgentChat, and Extensions APIs for flexible development.
    *   **Core API:** Core message passing, event-driven agents, and local/distributed runtime.
    *   **AgentChat API:** Rapid prototyping with common multi-agent patterns.
    *   **Extensions API:**  First- and third-party extensions (e.g., OpenAI, AzureOpenAI).
*   **Developer Tools:**
    *   [AutoGen Studio](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html): No-code GUI for building multi-agent applications.
    *   [AutoGen Bench](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html): Benchmarking suite for evaluating agent performance.
*   **Real-World Applications:**  Examples include [Magentic-One](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html), a multi-agent team for web browsing and code execution.
*   **Community & Support:** Join our active community through [Discord](https://aka.ms/autogen-discord), [GitHub Discussions](https://github.com/microsoft/autogen/discussions), [blog](https://devblogs.microsoft.com/autogen/), and weekly office hours.

## Where to Go Next

*   **Python:**  [Python Documentation](https://microsoft.github.io/autogen/stable/reference/index.html#)
*   **.NET:**  [.NET Documentation](https://microsoft.github.io/autogen/dotnet/dev/core/index.html)
*   **AutoGen Studio:** [AutoGen Studio Documentation](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/usage.html)

[View the AutoGen Packages on PyPi](https://pypi.org/search/?q=autogen)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## FAQ and Support

*   [Frequently Asked Questions (FAQ)](./FAQ.md)
*   [GitHub Discussions](https://github.com/microsoft/autogen/discussions)
*   [Discord Server](https://aka.ms/autogen-discord)
*   [Blog](https://devblogs.microsoft.com/autogen/)

## Legal Notices

*   [License](LICENSE) - MIT License
*   [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode)
*   [Privacy Information](https://go.microsoft.com/fwlink/?LinkId=521839)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>