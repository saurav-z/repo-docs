# IntentKit: Build Autonomous AI Agents with Ease

**Unlock the power of AI agents with IntentKit, a framework designed for creating, managing, and deploying intelligent agents capable of diverse tasks.**  [View the original repository](https://github.com/crestalnetwork/intentkit).

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to build sophisticated AI agents with features including blockchain interaction, social media management, and custom skill integration.

## Key Features of IntentKit

*   **Multiple Agent Support:** Easily manage and orchestrate several AI agents.
*   **Autonomous Agent Management:**  Provides the tools you need to handle autonomous behavior.
*   **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   **Extensible Skill System:** Customize agents with a wide range of skills.
*   **MCP (WIP):** More features in development!

## Architecture Overview

IntentKit's architecture is built to facilitate complex AI agent interactions, including integration with various platforms and services.

```
                                                                                    
                                 Entrypoints                                        
                       │                             │                              
                       │   Twitter/Telegram & more   │                              
                       └──────────────┬──────────────┘                              
                                      │                                             
  Storage:  ────┐                     │                      ┌──── Skills:          
                │                     │                      │                      
  Agent Config  │     ┌───────────────▼────────────────┐     │  Chain Integration   
                │     │                                │     │                      
  Credentials   │     │                                │     │  Wallet Management   
                │     │           The Agent            │     │                      
  Personality   │     │                                │     │  On-Chain Actions    
                │     │                                │     │                      
  Memory        │     │      Powered by LangGraph      │     │  Internet Search     
                │     │                                │     │                      
  Skill State   │     └────────────────────────────────┘     │  Image Processing    
            ────┘                                            └────                  
                                                                                    
                                                                More and More...    
                         ┌──────────────────────────┐                               
                         │                          │                               
                         │  Agent Config & Memory   │                               
                         │                          │                               
                         └──────────────────────────┘                               
                                                                                    
```

For in-depth information, please refer to the [Architecture](docs/architecture.md) documentation.

## Development & Usage

**Getting Started:**

1.  **Migration Note:** Migrate to uv from poetry: `rm -rf .venv` followed by `uv sync`.
2.  Follow the [Development Guide](DEVELOPMENT.md) to set up your development environment.
3.  Consult the [Documentation](docs/) for detailed instructions and usage examples.

## Project Structure

The project comprises a core package and an application, divided into the following key directories:

*   **`intentkit/`**: The IntentKit package.
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for external services.
    *   `config/`: System-level configurations.
    *   `core/`: Core agent system, powered by LangGraph.
    *   `models/`: Entity models (Pydantic and SQLAlchemy).
    *   `skills/`: Extensible skills system.
    *   `utils/`: Utility functions.
*   **`app/`**: The IntentKit application.
    *   `admin/`: Admin APIs and agent generation.
    *   `entrypoints/`: Agent interaction entrypoints (web, Telegram, etc.).
    *   `services/`: Service implementations (Telegram, Twitter, etc.).
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking logic.
    *   `readonly.py`: Read-only entrypoint.
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.
*   `docs/`: Documentation.
*   `scripts/`: Management and migration scripts.

## Agent API

IntentKit offers a comprehensive REST API for programmatic agent interaction.  This enables you to integrate agents with other systems and build custom interfaces.

*   **Agent API Documentation:** Access the [Agent API Documentation](docs/agent_api.md) to get started.

## Contributing

Contributions are welcome!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for open requests.
2.  Review the [Skill Development Guide](docs/contributing/skills.md) for guidance.

### Developer Chat

Join our [Discord](https://discord.com/invite/crestal) to connect with the developer community. Apply for the IntentKit dev role for advanced access.

## License

IntentKit is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.