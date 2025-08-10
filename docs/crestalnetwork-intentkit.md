# IntentKit: Build and Manage Autonomous AI Agents

**Unleash the power of AI with IntentKit, an open-source framework for creating and orchestrating intelligent agents capable of complex tasks.**  [View the original repository on GitHub](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to build and deploy AI agents with diverse capabilities, including blockchain interaction, social media management, and custom skill integration. This framework streamlines the development process, providing a robust foundation for creating intelligent, autonomous systems.

## Key Features of IntentKit

*   🤖 **Multi-Agent Support:** Manage and deploy multiple AI agents, each with unique roles and responsibilities.
*   🔄 **Autonomous Agent Management:**  Orchestrate and control your agents' behavior with built-in autonomous management features.
*   🔗 **Blockchain Integration:** Seamlessly interact with EVM-compatible blockchain networks for transactions and data access.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram to manage social media presence and engage with users.
*   🛠️ **Extensible Skill System:**  Customize your agents' capabilities with a modular skill system, allowing for easy integration of new functionalities.
*   🔌 **MCP (WIP):**  [Placeholder - Keep this.  If you know what this is replace with the relevant info.]

## Architecture Overview

IntentKit's architecture is designed for flexibility and scalability. Agents are built on LangGraph and interact with various entrypoints (Twitter, Telegram, etc.).  They leverage a combination of storage (Agent Config, Credentials, Memory), skills (Chain Integration, Internet Search, Image Processing), and core functionalities.  See the [Architecture](docs/architecture.md) section for more in-depth details.

## Getting Started

**1. Environment Setup (Important - Update Your Environment!)**

We've migrated to `uv` from `poetry`.  Follow these steps:

```bash
rm -rf .venv  # Delete the existing virtual environment
uv sync       # Create a new virtual environment
```

**2. Development:**

*   Consult the [Development Guide](DEVELOPMENT.md) for setting up your development environment.

**3. Documentation:**

*   Explore the comprehensive [Documentation](docs/) to learn more about IntentKit's features and capabilities.
*   Access the [Agent API Documentation](docs/agent_api.md) to programmatically interact with your agents.

## Project Structure

The project is organized into the `intentkit/` package and the `app/` application, along with documentation and supporting scripts:

**intentkit/ (Core Package)**

*   `abstracts/`: Abstract classes and interfaces.
*   `clients/`: Clients for external services.
*   `config/`: System-level configurations.
*   `core/`: The core agent system.
*   `models/`: Entity models.
*   `skills/`: Extensible skills system.
*   `utils/`: Utility functions.

**app/ (Application)**

*   `admin/`: Admin APIs and agent generators.
*   `entrypoints/`: Entrypoints for agent interaction (web, Telegram, Twitter).
*   `services/`: Service implementations (Telegram, Twitter, etc.).
*   `api.py`: REST API server.
*   `autonomous.py`: Autonomous agent runner.
*   `checker.py`: Health and credit checking logic.
*   `readonly.py`: Readonly entrypoint.
*   `scheduler.py`: Background task scheduler.
*   `singleton.py`: Singleton agent manager.
*   `telegram.py`: Telegram integration.
*   `twitter.py`: Twitter integration.

*   `docs/`: Documentation
*   `scripts/`: Operation and management scripts.

## Contributing

We welcome contributions! Please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

**Contribute Skills:**

1.  Check the [Wishlist](docs/contributing/wishlist.md) for active requests.
2.  Follow the [Skill Development Guide](docs/contributing/skills.md) for detailed instructions.

**Developer Community:**

*   Join our [Discord](https://discord.com/invite/crestal) for discussions and support. Apply for a dev role to access the dedicated developer channel.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.