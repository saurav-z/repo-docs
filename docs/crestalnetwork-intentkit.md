# IntentKit: Build Autonomous AI Agents with Ease

**Empower your projects with intelligent, self-managing AI agents capable of interacting with blockchains, social media, and more.** ([View the Original Repo](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an open-source framework designed for creating and managing autonomous AI agents. It provides a robust foundation for building agents that can perform complex tasks across various platforms and services.

## Key Features:

*   🤖 **Multi-Agent Support:** Create and manage multiple AI agents.
*   🔄 **Autonomous Agent Management:**  Effortlessly control and monitor your agents' behavior.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains, enabling on-chain actions.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram for automated social interaction.
*   🛠️ **Extensible Skill System:**  Easily integrate custom skills to expand agent capabilities.
*   🔌 **MCP (WIP):** Coming Soon!

## Why Use IntentKit?

*   **Simplified Agent Development:** Focus on building agent logic, not infrastructure.
*   **Rapid Prototyping:** Quickly deploy AI agents with pre-built integrations.
*   **Extensibility:**  Customize agents with a growing library of skills and integrations.

## Architecture

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

For a detailed view, see the [Architecture](docs/architecture.md) section.

## Getting Started

### Package Manager Migration Warning

*   **Important:**  We have migrated from Poetry to uv.
    *   Remove the existing virtual environment: `rm -rf .venv`
    *   Create a new environment: `uv sync` (This is a one-time setup)

### Development

*   Follow the [Development Guide](DEVELOPMENT.md) for setup instructions.
*   Consult the [Documentation](docs/) for detailed information and tutorials.

## Project Structure

*   **[intentkit/](intentkit/)**: The core Python package.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System configurations.
    *   [core/](intentkit/core/): Core agent system (LangGraph based).
    *   [models/](intentkit/models/): Entity models (Pydantic and SQLAlchemy).
    *   [skills/](intentkit/skills/): Extensible skill system (LangChain tools).
    *   [utils/](intentkit/utils/): Utility functions.

*   **[app/](app/)**:  The IntentKit application.
    *   [admin/](app/admin/): Admin APIs and agent generation.
    *   [entrypoints/](app/entrypoints/): Agent interaction entrypoints (web, Telegram, Twitter, etc.).
    *   [services/](app/services/): Service implementations (Telegram, Twitter, etc.).
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking.
    *   [readonly.py](app/readonly.py): Readonly entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Management scripts

## Agent API

**Programmatically control your agents using our comprehensive REST API.**

*   Explore the [Agent API Documentation](docs/agent_api.md).

## Contributing

**We welcome contributions!**

*   Read our [Contributing Guidelines](CONTRIBUTING.md).
*   Explore the [Wishlist](docs/contributing/wishlist.md) for feature requests.
*   See the [Skill Development Guide](docs/contributing/skills.md) to contribute new skills.

### Developer Community

*   Join our [Discord](https://discord.com/invite/crestal) (apply for dev role).

## License

This project is licensed under the [MIT License](LICENSE).