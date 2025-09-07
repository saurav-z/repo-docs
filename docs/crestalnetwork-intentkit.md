# IntentKit: Build and Manage Autonomous AI Agents

**Unleash the power of AI with IntentKit, a cutting-edge framework for creating and deploying autonomous AI agents with diverse capabilities.** ([View on GitHub](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to build and manage intelligent agents that can interact with the world, from blockchain and social media to custom skill integrations.

## Key Features

*   **🤖 Multiple Agent Support:** Manage and orchestrate multiple AI agents simultaneously.
*   **🔄 Autonomous Agent Management:**  Automated agent lifecycle management, from creation to operation.
*   **🔗 Blockchain Integration:** Seamlessly interact with EVM-compatible blockchain networks.
*   **🐦 Social Media Integration:** Engage with users on platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:** Easily integrate custom skills and functionalities using LangChain tools.
*   **🔌 MCP (WIP):** (Mentioning the feature and that it is work in progress.)

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility.  The core leverages LangGraph to provide a robust foundation for agent behavior.

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

For a more detailed understanding of the system's design, refer to the [Architecture](docs/architecture.md) documentation.

## Project Structure

The project is organized into the core `intentkit` package and the `app` application, along with supporting directories for documentation and scripts.

**Key directories:**

*   **[intentkit/](intentkit/)**:  The main IntentKit package. Contains core logic and modules.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System configurations.
    *   [core/](intentkit/core/): Core agent system.
    *   [models/](intentkit/models/): Entity models.
    *   [skills/](intentkit/skills/): Extensible skills system.
    *   [utils/](intentkit/utils/): Utility functions.
*   **[app/](app/)**: The IntentKit application (API server, runner, scheduler).
    *   [admin/](app/admin/): Admin APIs.
    *   [entrypoints/](app/entrypoints/): Agent interaction entrypoints.
    *   [services/](app/services/): Service implementations.
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking logic.
    *   [readonly.py](app/readonly.py): Readonly entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.
*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Operation and management scripts

## Agent API

Programmatically interact with your agents using IntentKit's comprehensive REST API.

**Explore the API:** [Agent API Documentation](docs/agent_api.md)

## Development and Contributing

### Getting Started

Refer to the [Development Guide](DEVELOPMENT.md) to set up your development environment.

### Documentation

Comprehensive documentation is available in the [Documentation](docs/) directory.

### Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for skill requests and then see the [Skill Development Guide](docs/contributing/skills.md).

### Developer Community

Join the discussion and connect with other developers:

*   [Discord](https://discord.com/invite/crestal)
*   Apply for an intentkit dev role in the Discord server.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Important Notes for Migration:**

If you are upgrading and using poetry, follow these steps:

1.  Delete the .venv folder: `rm -rf .venv`
2.  Run `uv sync` to create a new virtual environment.