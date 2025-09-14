# IntentKit: Build and Manage Powerful AI Agents

**Unleash the power of autonomous AI agents with IntentKit, a framework designed for seamless integration and dynamic capabilities.**

[Visit the original repository on GitHub](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to create and manage AI agents with a wide range of functionalities, including:

**Key Features:**

*   🤖 **Multi-Agent Support:** Manage multiple agents simultaneously.
*   🔄 **Autonomous Agent Management:** Control and orchestrate agent behaviors.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchain networks.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Integrate custom skills and expand agent capabilities.
*   🔌 **MCP (WIP):**  (Mention of future development)

## Architecture

IntentKit's architecture is designed for flexibility and extensibility. Key components include:

*   **Entrypoints:**  Handles input from various sources like Twitter and Telegram.
*   **Agent:** The core of the system, powered by LangGraph, managing agent logic and interactions.
*   **Skills:**  Enable the agent to perform various actions, such as blockchain interactions, internet searches, and image processing.
*   **Storage:** Manages agent configuration, credentials, personality, memory, and skill states.

For a more detailed understanding, refer to the [Architecture](docs/architecture.md) section in the documentation.

## Getting Started

### Package Manager Migration Warning

**Important:** If you are migrating from poetry, you need to delete the `.venv` folder and run `uv sync` to create a new virtual environment.

```bash
rm -rf .venv
uv sync
```

### Development

*   **Development Guide:** Get started with your setup by reading the [Development Guide](DEVELOPMENT.md).
*   **Documentation:** Explore the comprehensive [Documentation](docs/) to learn more about IntentKit's features and usage.

## Project Structure

IntentKit is organized into two primary components:

*   **[intentkit/](intentkit/)**: The core IntentKit package, available as a pip package.  Includes:
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for external services.
    *   `config/`: System configurations.
    *   `core/`: Core agent system.
    *   `models/`: Data models.
    *   `skills/`: The extensible skills system.
    *   `utils/`: Utility functions.
*   **[app/](app/)**: The IntentKit application (API server, autonomous runner, and scheduler).  Includes:
    *   `admin/`: Admin APIs and agent generators.
    *   `entrypoints/`: Interaction entrypoints (web, Telegram, Twitter).
    *   `services/`: Service implementations for various platforms.
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking logic.
    *   `readonly.py`: Readonly entrypoint.
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.

## Agent API

IntentKit provides a robust REST API for interacting with and managing your agents programmatically.  This allows you to build custom applications and integrate IntentKit with existing systems.

*   **Agent API Documentation:** Explore the [Agent API Documentation](docs/agent_api.md) to learn how to use the API.

## Contributing

We welcome contributions to IntentKit!

*   **Contributing Guidelines:** Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.
*   **Contribute Skills:**
    *   Check the [Wishlist](docs/contributing/wishlist.md) for active requests.
    *   Refer to the [Skill Development Guide](docs/contributing/skills.md) to learn how to create and contribute skills.

### Developer Community

*   **Discord:** Join our [Discord](https://discord.com/invite/crestal) to connect with the development community.  Apply for an IntentKit dev role by opening a support ticket.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.