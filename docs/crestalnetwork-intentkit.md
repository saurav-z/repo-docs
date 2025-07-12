# IntentKit: Build Autonomous AI Agents with Ease

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

**IntentKit** is an open-source framework empowering developers to create and manage powerful AI agents capable of interacting with blockchain, social media, and custom skills.  

[View the original repository on GitHub](https://github.com/crestalnetwork/intentkit)

## Key Features

*   **Multiple Agent Support:** Manage and orchestrate multiple AI agents simultaneously.
*   **Autonomous Agent Management:** Design agents with self-governing capabilities.
*   **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   **Social Media Integration:** Connect to platforms like Twitter and Telegram.
*   **Extensible Skill System:** Easily add new skills and functionalities to your agents.
*   **MCP (WIP):** (Mention of future functionality)

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

For a more in-depth look at the architecture, please refer to the [Architecture](docs/architecture.md) section.

## Getting Started

### Package Manager Migration Warning

If you're updating from a previous version, you'll need to migrate to the new package manager:

1.  Delete the existing virtual environment: `rm -rf .venv`
2.  Run: `uv sync` to create a new virtual environment.

## Development

To begin developing with IntentKit, consult the [Development Guide](DEVELOPMENT.md).

## Documentation

Explore the comprehensive documentation for detailed information and usage examples in the [Documentation](docs/) directory.

## Project Structure

*   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
*   [app/](app/): Core application code
    *   [core/](intentkit/core/): Core modules
    *   [services/](app/services/): Services
    *   [entrypoints/](app/entrypoints/): Entrypoints means the way to interact with the agent
    *   [admin/](app/admin/): Admin logic
    *   [config/](intentkit/config/): Configurations
    *   [api.py](app/api.py): REST API server
    *   [autonomous.py](app/autonomous.py): Autonomous agent scheduler
    *   [singleton.py](app/singleton.py): Singleton agent scheduler
    *   [scheduler.py](app/scheduler.py): Scheduler for periodic tasks
    *   [readonly.py](app/readonly.py): Readonly entrypoint
    *   [telegram.py](app/telegram.py): Telegram listener
*   [clients/](intentkit/clients/): Clients for external services
*   [docs/](docs/): Documentation
*   [models/](intentkit/models/): Database models
*   [scripts/](scripts/): Scripts for agent management
*   [skills/](intentkit/skills/): Skill implementations
*   [utils/](intentkit/utils/): Utility functions

## Contributing

Contributions are highly encouraged!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for active requests.
2.  See the [Skill Development Guide](docs/contributing/skills.md) for detailed instructions.

### Developer Community

Join the IntentKit developer community on [Discord](https://discord.com/invite/crestal).  Apply for an IntentKit dev role by opening a support ticket there. We have a discussion channel for you to connect with the rest of the developers.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for more details.