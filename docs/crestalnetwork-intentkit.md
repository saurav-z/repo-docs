# IntentKit: Build and Manage Autonomous AI Agents with Ease

**Unleash the power of AI agents with IntentKit, a cutting-edge framework for creating, managing, and deploying intelligent autonomous systems.**  [View the original repository](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is designed to empower developers to build AI agents with diverse capabilities, from interacting with blockchains to managing social media and integrating custom skills.

## Key Features

*   **🤖 Multi-Agent Support:** Manage and orchestrate multiple autonomous agents simultaneously.
*   **🔄 Autonomous Agent Management:** Streamlined framework for creating, deploying, and maintaining AI agents.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains for decentralized applications.
*   **🐦 Social Media Integration:** Connect with and manage content on platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:** Easily integrate custom skills and functionalities to expand agent capabilities.
*   **🔌 MCP (WIP):** (Placeholder, specific details in the original README)

## Architecture Overview

IntentKit utilizes a modular architecture, leveraging LangGraph for core agent functionality. This design supports seamless integration of various inputs, storage solutions, and skills.

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

For a more in-depth understanding of the system architecture, please refer to the [Architecture](docs/architecture.md) documentation.

## Getting Started

### Development

Follow the [Development Guide](DEVELOPMENT.md) to set up your development environment.

### Documentation

Consult the comprehensive [Documentation](docs/) for detailed information and usage examples.

## Project Structure

The project is organized into distinct modules:

*   **[intentkit/](intentkit/)**: The core IntentKit package.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System configurations
    *   [core/](intentkit/core/): Core agent system
    *   [models/](intentkit/models/): Entity models
    *   [skills/](intentkit/skills/): Extensible skills system
    *   [utils/](intentkit/utils/): Utility functions
*   **[app/](app/)**: The IntentKit application.
    *   [admin/](app/admin/): Admin APIs and agent generators
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
*   [scripts/](scripts/): Management and migration scripts

## Agent API

IntentKit provides a powerful REST API for programmatic control of your AI agents.  Build custom applications, integrate with existing systems, and create tailored interfaces using the Agent API.

**Explore the API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests.
2.  Follow the [Skill Development Guide](docs/contributing/skills.md) to start building your own skills.

### Developer Community

Join our [Discord](https://discord.com/invite/crestal) and request an intentkit dev role for collaborative development and support.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.