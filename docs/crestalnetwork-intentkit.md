# IntentKit: Build and Manage Autonomous AI Agents

**Unleash the power of autonomous AI with IntentKit, a cutting-edge framework for creating, managing, and deploying intelligent agents with diverse capabilities.** [(View Original Repo)](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to build sophisticated AI agents capable of interacting with the blockchain, managing social media, and integrating custom skills. This framework simplifies the creation and deployment of autonomous agents, offering a robust and extensible platform.

## Key Features

*   **🤖 Multiple Agent Support:** Manage a fleet of AI agents, each with its own personality, skills, and goals.
*   **🔄 Autonomous Agent Management:** Design and deploy agents that can operate independently, making decisions and taking actions based on their programming.
*   **🔗 Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains, enabling on-chain actions and data retrieval.
*   **🐦 Social Media Integration:** Connect with platforms like Twitter and Telegram to manage social media presence and engage with audiences.
*   **🛠️ Extensible Skill System:** Easily add new skills and functionalities to your agents using a modular and adaptable system.
*   **🔌 MCP (WIP):** (More details coming soon)

## Architecture

IntentKit's architecture is designed for flexibility and scalability. Agents are powered by LangGraph and leverage a combination of core components, skills, and integrations to achieve their objectives.

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

For a deeper understanding of the system's design, explore the [Architecture](docs/architecture.md) section in the documentation.

## Development

Get up and running quickly!  See the [Development Guide](DEVELOPMENT.md) to set up your environment and start building.

## Documentation

Comprehensive documentation is available to guide you through every step. Refer to the [Documentation](docs/) for detailed information.

## Project Structure

The project is divided into modular components for ease of development and maintenance:

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

**Programmatically Interact with Your Agents:**  IntentKit provides a robust REST API for seamless integration and control.

**Explore the API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contribute Skills

Extend IntentKit's capabilities by contributing new skills.

1.  Check the [Wishlist](docs/contributing/wishlist.md) for open requests.
2.  Review the [Skill Development Guide](docs/contributing/skills.md) for detailed instructions.

### Developer Chat

Join our community and connect with other developers!  Apply for an IntentKit dev role and start contributing to our [Discord](https://discord.com/invite/crestal) server.

## License

This project is licensed under the [MIT License](LICENSE).