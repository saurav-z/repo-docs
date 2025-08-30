# IntentKit: Build Autonomous AI Agents with Ease

**Unleash the power of autonomous AI agents for blockchain interaction, social media management, and beyond with IntentKit.** [(View on GitHub)](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an open-source framework designed to simplify the creation and management of AI agents.  It offers a modular and extensible architecture, allowing you to build agents with a wide range of capabilities, from interacting with blockchains to managing your social media presence.

## Key Features

*   🤖 **Multi-Agent Support:** Manage and orchestrate multiple autonomous agents within a single framework.
*   🔄 **Autonomous Agent Management:**  Built-in features for agent lifecycle management and execution.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains for transactions, data retrieval, and more.
*   🐦 **Social Media Integration:**  Connect to platforms like Twitter and Telegram to automate content creation, engagement, and monitoring.
*   🛠️ **Extensible Skill System:**  Easily add new skills and capabilities to your agents using a flexible, LangChain-based system.
*   🔌 **MCP (WIP):** Explore the Multi-Chain Protocol in the works.

## Architecture Overview

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

For a deeper dive into the system's design, refer to the detailed [Architecture](docs/architecture.md) documentation.

## Getting Started

### Package Manager Migration

**Important:** The project has migrated to `uv` from `poetry`. To set up your environment:

```bash
rm -rf .venv  # Remove existing virtual environment
uv sync       # Create a new virtual environment using uv
```

### Development

Start building and contributing! Consult the [Development Guide](DEVELOPMENT.md) for setup instructions.

## Documentation

Comprehensive documentation is available to help you get the most out of IntentKit.

*   Explore the complete [Documentation](docs/) for detailed guides and API references.
*   Access the [Agent API Documentation](docs/agent_api.md) to integrate your agents programmatically.

## Project Structure

The project is organized into distinct modules for clarity and maintainability:

*   **[intentkit/](intentkit/)**:  The core IntentKit package.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for interacting with external services.
    *   [config/](intentkit/config/): System-level configuration settings.
    *   [core/](intentkit/core/): The core agent system, powered by LangGraph.
    *   [models/](intentkit/models/): Data models using Pydantic and SQLAlchemy.
    *   [skills/](intentkit/skills/): The extensible skill system, based on LangChain tools.
    *   [utils/](intentkit/utils/): Utility functions.
*   **[app/](app/)**: The IntentKit application.
    *   [admin/](app/admin/): Admin APIs and agent management tools.
    *   [entrypoints/](app/entrypoints/): Entrypoints for interacting with agents.
    *   [services/](app/services/): Service implementations for various platforms.
    *   [api.py](app/api.py): REST API server for agent interaction.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking logic.
    *   [readonly.py](app/readonly.py): Read-only entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.
*   [docs/](docs/): Documentation.
*   [scripts/](scripts/): Utility scripts for management and migrations.

## Contributing

We welcome contributions!

*   Read the [Contributing Guidelines](CONTRIBUTING.md) for instructions.
*   Check the [Wishlist](docs/contributing/wishlist.md) to see actively requested skills.
*   Learn how to develop skills with the [Skill Development Guide](docs/contributing/skills.md).

### Developer Community

Join the IntentKit community!
*   Connect with developers on [Discord](https://discord.com/invite/crestal) and request an IntentKit dev role.
*   Collaborate with the team in the discussion channel.

## License

IntentKit is licensed under the [MIT License](LICENSE).