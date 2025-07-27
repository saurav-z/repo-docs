# IntentKit: Build and Manage Powerful AI Agents

**IntentKit** is a cutting-edge autonomous agent framework designed to empower developers to create and manage sophisticated AI agents with diverse capabilities. [Explore the original repository](https://github.com/crestalnetwork/intentkit).

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   🤖 **Multi-Agent Support:** Manage and deploy multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:** Orchestrate and control your agents with ease.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchain networks.
*   🐦 **Social Media Integration:** Connect agents to platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Customize your agents with a flexible skill architecture.
*   🔌 **MCP (WIP):** [Placeholder for the Multi-Chain Protocol (WIP) - Please update]

## Architecture Overview

IntentKit utilizes a modular architecture, allowing for seamless integration of various functionalities. The system is built around the core agent, which interacts with various entrypoints (e.g., Twitter, Telegram) and utilizes skills for tasks such as blockchain interaction, social media management, and internet searches.

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

For a detailed understanding of the architecture, refer to the [Architecture](docs/architecture.md) section.

## Development

Get started by following the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to guide you through the framework. Explore the [Documentation](docs/) for more details.

## Project Structure

The project is organized into two main components:

*   **[intentkit/](intentkit/)**: The IntentKit package (pip installable)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System level configurations
    *   [core/](intentkit/core/): Core agent system
    *   [models/](intentkit/models/): Entity models
    *   [skills/](intentkit/skills/): Extensible skills system
    *   [utils/](intentkit/utils/): Utility functions

*   **[app/](app/)**: The IntentKit application
    *   [admin/](app/admin/): Admin APIs and related functionality
    *   [entrypoints/](app/entrypoints/): Entrypoints for agent interaction
    *   [services/](app/services/): Service implementations
    *   [api.py](app/api.py): REST API server
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner
    *   [checker.py](app/checker.py): Health and credit checking logic
    *   [readonly.py](app/readonly.py): Readonly entrypoint
    *   [scheduler.py](app/scheduler.py): Background task scheduler
    *   [singleton.py](app/singleton.py): Singleton agent manager
    *   [telegram.py](app/telegram.py): Telegram integration
    *   [twitter.py](app/twitter.py): Twitter integration

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Management scripts

## Agent API

Leverage the powerful REST API to programmatically access and interact with your agents.

*   **Agent API Documentation:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

Refer to the [Wishlist](docs/contributing/wishlist.md) for active requests and the [Skill Development Guide](docs/contributing/skills.md) for detailed instructions.

### Developer Chat

Join our Discord community for discussions and support! Apply for the intentkit dev role in the [Discord](https://discord.com/invite/crestal) to participate in the developer discussion channel.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Package Manager Migration Warning

We just migrated to uv from poetry.
You need to delete the .venv folder and run `uv sync` to create a new virtual environment. (one time)
```bash
rm -rf .venv
uv sync
```