# IntentKit: Build Powerful AI Agents for Automation and Interaction

**Unlock the power of autonomous AI with IntentKit, a cutting-edge framework for creating and managing intelligent agents.** ([View on GitHub](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is designed to enable the rapid development and deployment of AI agents capable of performing complex tasks across various platforms and services. Whether you're looking to automate blockchain interactions, manage social media, or integrate custom skills, IntentKit provides the tools you need.

## Key Features

*   **🤖 Multiple Agent Support:** Manage and orchestrate multiple AI agents within a single framework.
*   **🔄 Autonomous Agent Management:**  Automate the lifecycle and operations of your AI agents.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchain networks.
*   **🐦 Social Media Integration:** Seamlessly manage and interact with platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:**  Add new capabilities and functionalities with a flexible skill architecture.
*   **🔌 MCP (WIP):**  (Mention what this is or remove if unclear)

## Architecture

IntentKit's architecture is designed for flexibility and scalability. The core components include:

*   **Entrypoints:** Interfaces for interacting with agents (e.g., Twitter, Telegram).
*   **Agent Core:** Powered by LangGraph, managing agent logic and interactions.
*   **Skills:**  Extend agent capabilities through integrations like Chain Integration, Internet Search, Image Processing, etc.
*   **Storage:** Agent Config, Credentials, Personality, Memory, Skill State.

For a more detailed understanding, explore the [Architecture](docs/architecture.md) documentation.

## Getting Started

### Package Manager Migration

**Important:**  We've migrated to `uv` from `poetry`.  Follow these steps to update your environment:

```bash
rm -rf .venv
uv sync
```

### Development

Begin your development journey with the [Development Guide](DEVELOPMENT.md) to set up your environment.

### Documentation

Comprehensive documentation is available to guide you through IntentKit's features and functionalities.  Start by reviewing the [Documentation](docs/).

## Project Structure

The project is structured into two main components:

*   **`intentkit/`:** The core IntentKit package (available via pip).
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for external services.
    *   `config/`: System-level configurations.
    *   `core/`: Core agent system.
    *   `models/`: Entity models.
    *   `skills/`: Extensible skills system.
    *   `utils/`: Utility functions.
*   **`app/`:** The IntentKit application (API server, autonomous runner, scheduler).
    *   `admin/`: Admin APIs and agent management.
    *   `entrypoints/`: Entrypoints for agent interaction.
    *   `services/`: Service implementations.
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking.
    *   `readonly.py`: Readonly entrypoint.
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.
*   `docs/`: Documentation
*   `scripts/`: Scripts for management and migrations

## Agent API

IntentKit offers a robust REST API, enabling seamless programmatic access and control over your AI agents.

**Explore the API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

Your contributions are welcome!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

*   First, check the [Wishlist](docs/contributing/wishlist.md) for active requests.
*   Then, consult the [Skill Development Guide](docs/contributing/skills.md) for implementation details.

### Developer Community

Connect with other developers and get support on our [Discord](https://discord.com/invite/crestal).  Apply for an intentkit dev role in the support ticket.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.