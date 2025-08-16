# IntentKit: Build Autonomous AI Agents with Ease

**Unlock the power of AI agents with IntentKit, a versatile framework designed for creating and managing intelligent systems.** ([Original Repo](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an open-source framework that empowers you to build and deploy AI agents capable of a wide range of tasks, from blockchain interaction to social media management and beyond.

## Key Features

*   **🤖 Multiple Agent Support:** Manage and deploy multiple AI agents simultaneously.
*   **🔄 Autonomous Agent Management:** Oversee the lifecycle and operations of your AI agents.
*   **🔗 Blockchain Integration:** Interact seamlessly with EVM-compatible blockchain networks.
*   **🐦 Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:** Easily integrate custom skills and functionalities.
*   **🔌 MCP (WIP):** (Mention of this feature. Consider removing this from the summary, or adding a full sentence if there is enough context.)

## Architecture Overview

IntentKit's architecture is built around a core agent system driven by LangGraph. Agents interact with various entrypoints (e.g., Twitter, Telegram) and leverage a range of skills and integrations. The system includes:

*   **Entrypoints:** Handles interactions with external platforms (e.g., Twitter, Telegram).
*   **Core Agent:** Powered by LangGraph, manages agent logic and workflows.
*   **Skills:** Provides specialized functionalities, including:
    *   Chain Integration (Wallet Management, On-Chain Actions)
    *   Internet Search
    *   Image Processing
*   **Storage:** Stores Agent Config, Credentials, Personality, Memory, Skill State
*   **Agent Config & Memory:** The configurations and memory of each agent

For a more detailed view, refer to the [Architecture](docs/architecture.md) documentation.

## Getting Started

*   **Development:** Explore the [Development Guide](DEVELOPMENT.md) to set up your development environment.
*   **Documentation:** Consult the comprehensive [Documentation](docs/) for in-depth information and usage examples.
*   **Agent API:** Use the REST API to interact with agents.  See the [Agent API Documentation](docs/agent_api.md).

## Project Structure

*   **[intentkit/](intentkit/)**: The core IntentKit package (pip installable)
*   **[app/](app/)**: The IntentKit application (API server, autonomous runner, scheduler)
*   **[docs/](docs/)**: Documentation
*   **[scripts/](scripts/)**: Scripts for management and migrations

## Contribute

We welcome contributions!

*   Read our [Contributing Guidelines](CONTRIBUTING.md)
*   Contribute to [Skill Development Guide](docs/contributing/skills.md) to create new skills, and view the [Wishlist](docs/contributing/wishlist.md) for active requests.
*   Join our [Discord](https://discord.com/invite/crestal) and connect with the development community.

## License

IntentKit is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---