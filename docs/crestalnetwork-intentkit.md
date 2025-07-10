# IntentKit: Build Autonomous AI Agents with Ease

[View the original repository on GitHub](https://github.com/crestalnetwork/intentkit)

IntentKit is a powerful, open-source framework for building and managing sophisticated AI agents capable of interacting with blockchains, social media, and custom skills.

## Key Features

*   **Multi-Agent Support:** Manage and orchestrate multiple AI agents simultaneously.
*   **Autonomous Agent Management:** Control and monitor agents with self-governing capabilities.
*   **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   **Extensible Skill System:** Easily add new skills to expand agent functionality.
*   **MCP (Work in Progress):**  (Placeholder - explain what MCP is when available)

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
  Personality   │     │                                │     │                      
  Memory        │     │      Powered by LangGraph      │     │  On-Chain Actions    
                │     │                                │     │                      
  Skill State   │     └────────────────────────────────┘     │  Internet Search     
            ────┘                                            └────                  
                                                                                    
                                                                More and More...    
                         ┌──────────────────────────┐                               
                         │                          │                               
                         │  Agent Config & Memory   │                               
                         │                          │                               
                         └──────────────────────────┘                               
                                                                                    
```

For a more detailed explanation, please refer to the [Architecture](docs/architecture.md) documentation.

## Getting Started

### Package Manager Migration Warning

**Important:** We have migrated to `uv` from `poetry`.  To set up your environment:

1.  Remove your existing virtual environment: `rm -rf .venv`
2.  Create a new virtual environment: `uv sync`

## Development

*   **Development Guide:** Read the [Development Guide](DEVELOPMENT.md) for setup instructions.
*   **Documentation:** Explore the comprehensive [Documentation](docs/) to understand the framework.

## Project Structure

*   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
*   [app/](app/): Core application code
    *   [core/](intentkit/core/): Core modules
    *   [services/](app/services/): Services
    *   [entrypoints/](app/entrypoints/): Entrypoints to interact with the agent
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

We welcome contributions! Please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for existing requests.
2.  Consult the [Skill Development Guide](docs/contributing/skills.md) for instructions.

### Developer Community

*   **Discord:** Join our [Discord](https://discord.com/invite/crestal) to connect with other developers and request a dev role for IntentKit discussions.