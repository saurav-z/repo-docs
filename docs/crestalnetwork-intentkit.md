# IntentKit: Build Autonomous AI Agents with Ease

**IntentKit empowers you to create and manage sophisticated AI agents capable of interacting with blockchains, social media, and custom integrations.** Explore the full capabilities on [GitHub](https://github.com/crestalnetwork/intentkit).

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   🤖 **Multi-Agent Support:** Manage multiple autonomous agents concurrently.
*   🔄 **Autonomous Agent Management:** Efficiently control and orchestrate your AI agents.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter, Telegram, and more.
*   🛠️ **Extensible Skill System:** Easily integrate custom functionalities and capabilities.
*   🔌 **MCP (WIP):**  Modular Component Protocol for advanced agent design.

## Architecture

IntentKit's architecture is designed for flexibility and extensibility.  The core of the system leverages LangGraph for agent management, integrating various components:

*   **Entrypoints:**  Handles interactions from various sources (Twitter, Telegram, etc.).
*   **Storage:** Manages agent configurations, credentials, personality, memory, and skill states.
*   **Skills:** Enables a broad range of agent capabilities, including:
    *   Chain Integration
    *   Wallet Management
    *   On-Chain Actions
    *   Internet Search
    *   Image Processing
    *   And more...

For a detailed architectural overview, refer to the [Architecture](docs/architecture.md) section.

## Development

Get started with IntentKit by following the [Development Guide](DEVELOPMENT.md).

## Documentation & Resources

*   **Comprehensive Documentation:** Explore the full documentation suite at [Documentation](docs/).
*   **Agent API:** Utilize the REST API for programmatic access to your agents via the [Agent API Documentation](docs/agent_api.md).

## Project Structure

*   **[intentkit/](intentkit/)**: The IntentKit Python package (installed via pip)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System configurations.
    *   [core/](intentkit/core/): Core agent system (LangGraph based).
    *   [models/](intentkit/models/): Data models using Pydantic and SQLAlchemy.
    *   [skills/](intentkit/skills/): Extensible skill system.
    *   [utils/](intentkit/utils/): Utility functions.
*   **[app/](app/)**: The IntentKit application (API server, autonomous runner, and background scheduler)
    *   [admin/](app/admin/): Admin APIs.
    *   [entrypoints/](app/entrypoints/): Entrypoints for interacting with agents.
    *   [services/](app/services/): Service implementations.
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking.
    *   [readonly.py](app/readonly.py): Readonly entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.
*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Operation and management scripts

## Contributing

Contributions are welcome! Please review our [Contributing Guidelines](CONTRIBUTING.md).

### Contributing Skills

Check the [Wishlist](docs/contributing/wishlist.md) for active requests. Learn more in the [Skill Development Guide](docs/contributing/skills.md).

### Developer Community

Join the community on [Discord](https://discord.com/invite/crestal) and apply for a developer role to participate in discussions.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.

## Package Manager Migration Warning

After migrating to uv from poetry, delete the .venv folder and run `uv sync` to create a new virtual environment. (one time)
```bash
rm -rf .venv
uv sync