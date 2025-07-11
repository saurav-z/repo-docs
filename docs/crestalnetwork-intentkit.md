# IntentKit: Build Powerful AI Agents for Web3 and Beyond

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an open-source framework empowering developers to create and manage sophisticated AI agents, seamlessly integrating with blockchain, social media, and custom functionalities. Explore the original repository on GitHub: [https://github.com/crestalnetwork/intentkit](https://github.com/crestalnetwork/intentkit)

## Key Features

*   **Multi-Agent Support:** Manage and orchestrate multiple AI agents simultaneously.
*   **Autonomous Agent Management:** Enable agents to operate and make decisions independently.
*   **Blockchain Integration:** Interact with EVM-compatible blockchains for on-chain actions.
*   **Social Media Integration:** Connect with platforms like Twitter and Telegram to manage social presence.
*   **Extensible Skill System:** Easily integrate custom skills and expand agent capabilities.
*   **MCP (WIP):**  (Master Control Program - Work in Progress)

## Architecture Overview

IntentKit's architecture provides a flexible foundation for building AI agents. The core components include:

*   **Entrypoints:**  Handles interactions with external sources like social media platforms (Twitter, Telegram).
*   **Agent Core:** Powered by LangGraph, manages agent configuration, memory, and interactions.
*   **Skills:**  Provide specific functionalities, including:
    *   Chain Integration (Blockchain interaction)
    *   Wallet Management
    *   On-Chain Actions
    *   Internet Search
    *   Image Processing
*   **Storage:** Stores Agent Config, Credentials, Personality, Memory, and Skill State.

For a more detailed architectural view, see the [Architecture](docs/architecture.md) section.

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
                │     │                                │     │  Image Processing    
  Skill State   │     └────────────────────────────────┘     │                      
            ────┘                                            └────                  
                                                                                    
                                                                More and More...    
                         ┌──────────────────────────┐                               
                         │                          │                               
                         │  Agent Config & Memory   │                               
                         │                          │                               
                         └──────────────────────────┘                               
                                                                                    
```

## Development

Get started with IntentKit by reading the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available in the [Documentation](docs/) directory.

## Project Structure

*   **[abstracts/](intentkit/abstracts/)**: Abstract classes and interfaces
*   **[app/](app/)**: Core application code
    *   **[core/](intentkit/core/)**: Core modules
    *   **[services/](app/services/)**: Services
    *   **[entrypoints/](app/entrypoints/)**: Entrypoints means the way to interact with the agent
    *   **[admin/](app/admin/)**: Admin logic
    *   **[config/](intentkit/config/)**: Configurations
    *   **[api.py](app/api.py)**: REST API server
    *   **[autonomous.py](app/autonomous.py)**: Autonomous agent scheduler
    *   **[singleton.py](app/singleton.py)**: Singleton agent scheduler
    *   **[scheduler.py](app/scheduler.py)**: Scheduler for periodic tasks
    *   **[readonly.py](app/readonly.py)**: Readonly entrypoint
    *   **[telegram.py](app/telegram.py)**: Telegram listener
*   **[clients/](intentkit/clients/)**: Clients for external services
*   **[docs/](docs/)**: Documentation
*   **[models/](intentkit/models/)**: Database models
*   **[scripts/](scripts/)**: Scripts for agent management
*   **[skills/](intentkit/skills/)**: Skill implementations
*   **[utils/](intentkit/utils/)**: Utility functions

## Contributing

Contributions are welcome! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests.

Refer to the [Skill Development Guide](docs/contributing/skills.md) for detailed instructions.

### Developer Chat

Join the IntentKit community on [Discord](https://discord.com/invite/crestal) and request an IntentKit developer role.  There is a dedicated discussion channel for developers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.