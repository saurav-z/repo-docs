# IntentKit: Build Autonomous AI Agents for a Connected World

**Unlock the power of autonomous AI with IntentKit, a versatile framework designed to create and manage intelligent agents capable of interacting with blockchain, social media, and more.** ([Original Repo](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to build sophisticated AI agents with a wide range of capabilities, from blockchain interaction to social media management and custom skill integration.

## Key Features

*   🤖 **Multiple Agent Support:** Manage and deploy numerous AI agents simultaneously.
*   🔄 **Autonomous Agent Management:**  Automate agent workflows and decision-making.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:**  Easily integrate custom skills and functionalities.
*   🔌 **MCP (WIP):**  [Placeholder:  Describe the Multi-Chain Protocol functionality when available.]

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility. The core agent system, powered by LangGraph, integrates various components to enable sophisticated AI agent behavior.

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

For a more detailed explanation, please refer to the [Architecture](docs/architecture.md) section.

## Development & Getting Started

### Package Manager Migration Warning

**Important:**  The project has migrated from Poetry to `uv`. Follow these steps to update your environment:

1.  Delete the existing virtual environment:  `rm -rf .venv`
2.  Create a new virtual environment using `uv`:  `uv sync`

### Resources

*   **Development:** Get started with your setup by reading the [Development Guide](DEVELOPMENT.md).
*   **Documentation:** Explore the comprehensive [Documentation](docs/) to learn more about IntentKit.
*   **Agent API:** Access and control your agents programmatically through the [Agent API Documentation](docs/agent_api.md).

## Project Structure

*   **[intentkit/](intentkit/)**: The core IntentKit package, including:
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System configurations
    *   [core/](intentkit/core/): Core agent system
    *   [models/](intentkit/models/): Entity models
    *   [skills/](intentkit/skills/): Extensible skills system
    *   [utils/](intentkit/utils/): Utility functions
*   **[app/](app/)**: The IntentKit application, including:
    *   [admin/](app/admin/): Admin APIs
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
*   [scripts/](scripts/): Operation and temporary scripts

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contribute Skills

*   Check the [Wishlist](docs/contributing/wishlist.md) for active requests.
*   Consult the [Skill Development Guide](docs/contributing/skills.md) for information on creating new skills.

### Developer Community

*   Join the discussion on our [Discord](https://discord.com/invite/crestal) and request the IntentKit dev role.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.