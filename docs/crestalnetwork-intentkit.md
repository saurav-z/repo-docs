# IntentKit: Build and Manage Autonomous AI Agents

**IntentKit empowers you to build and manage sophisticated AI agents with blockchain interaction, social media integration, and custom skill capabilities.** Check out the [original repo](https://github.com/crestalnetwork/intentkit) for the source code and more details.

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   **Multiple Agent Support:** Manage and deploy multiple AI agents.
*   **Autonomous Agent Management:** Orchestrate and oversee agent operations seamlessly.
*   **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   **Extensible Skill System:** Customize agents with a modular skill system based on LangChain tools.
*   **MCP (Work in Progress):** Exploring multi-chain platform functionality.

## Architecture Overview

IntentKit's architecture leverages LangGraph to create powerful, autonomous agents. Here's a simplified view:

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

For a deeper dive into the architecture, see the [Architecture](docs/architecture.md) documentation.

## Getting Started

### Prerequisites
You will need to have `uv` and `python` installed on your system.

1.  **Migrate Environment (One-Time Setup):**

    ```bash
    rm -rf .venv
    uv sync
    ```

2.  **Development:**
    Familiarize yourself with the setup by reading the [Development Guide](DEVELOPMENT.md).

3.  **Documentation:**
    Explore the comprehensive [Documentation](docs/) for in-depth information.

## Project Structure

The project is organized into core functionality and the application layer:

*   **[intentkit/](intentkit/)**: The IntentKit package (published as a pip package)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces for core and skills
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System level configurations
    *   [core/](intentkit/core/): Core agent system, driven by LangGraph
    *   [models/](intentkit/models/): Entity models using Pydantic and SQLAlchemy
    *   [skills/](intentkit/skills/): Extensible skills system, based on LangChain tools
    *   [utils/](intentkit/utils/): Utility functions

*   **[app/](app/)**: The IntentKit app (API server, autonomous runner, and background scheduler)
    *   [admin/](app/admin/): Admin APIs, agent generators, and related functionality
    *   [entrypoints/](app/entrypoints/): Entrypoints for interacting with agents (web, Telegram, Twitter, etc.)
    *   [services/](app/services/): Service implementations for Telegram, Twitter, etc.
    *   [api.py](app/api.py): REST API server
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner
    *   [checker.py](app/checker.py): Health and credit checking logic
    *   [readonly.py](app/readonly.py): Readonly entrypoint
    *   [scheduler.py](app/scheduler.py): Background task scheduler
    *   [singleton.py](app/singleton.py): Singleton agent manager
    *   [telegram.py](app/telegram.py): Telegram integration
    *   [twitter.py](app/twitter.py): Twitter integration

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Operation and temporary scripts for management and migrations

## Agent API

Access and control your agents programmatically with the IntentKit REST API.

**Explore the API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

*   Check the [Wishlist](docs/contributing/wishlist.md) for actively requested skills.
*   Follow the [Skill Development Guide](docs/contributing/skills.md) to start building.

### Developer Community

Join the conversation on our [Discord](https://discord.com/invite/crestal) and request an intentkit dev role.  We have a discussion channel for developers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.