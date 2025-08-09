# IntentKit: Build and Manage Autonomous AI Agents with Ease

**Unlock the power of AI with IntentKit, a versatile framework for creating and deploying intelligent agents that interact with the blockchain, social media, and more.** [Explore the Original Repo](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit provides a robust foundation for developing and managing AI agents with diverse capabilities. Whether you're interested in blockchain interactions, social media automation, or custom skill integrations, IntentKit offers the tools you need to succeed.

## Key Features

*   **🤖 Multiple Agent Support:** Manage and deploy numerous autonomous agents simultaneously.
*   **🔄 Autonomous Agent Management:** Streamline agent lifecycles, including creation, monitoring, and updates.
*   **🔗 Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains for on-chain actions.
*   **🐦 Social Media Integration:** Connect with platforms like Twitter and Telegram for automated content management and interaction.
*   **🛠️ Extensible Skill System:** Customize agent capabilities through a flexible skill architecture, using LangChain tools.
*   **🔌 MCP (WIP):**  (Details coming soon!)

## Architecture Overview

IntentKit's architecture is designed for scalability and extensibility. Agents are driven by LangGraph and leverage various components for interacting with external services and managing their internal state.

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

For a deeper dive, refer to the [Architecture](docs/architecture.md) documentation.

## Getting Started

### Development

Get your development environment set up by following the instructions in the [Development Guide](DEVELOPMENT.md).

### Documentation

Explore the comprehensive [Documentation](docs/) to learn about IntentKit's features and capabilities.

## Project Structure

The project is organized into the `intentkit` package and the `app` application:

*   **`intentkit/`:** The core IntentKit package (installable via pip)
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for external services.
    *   `config/`: System-level configurations.
    *   `core/`: Core agent system (powered by LangGraph).
    *   `models/`: Data models using Pydantic and SQLAlchemy.
    *   `skills/`: Extensible skill system (based on LangChain tools).
    *   `utils/`: Utility functions.
*   **`app/`:** The IntentKit application (API server, autonomous runner, and scheduler)
    *   `admin/`: Admin APIs and agent management tools.
    *   `entrypoints/`: Entrypoints for agent interaction (web, Telegram, Twitter, etc.).
    *   `services/`: Service implementations (Telegram, Twitter, etc.).
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking logic.
    *   `readonly.py`: Read-only entrypoint.
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.
*   `docs/`: Documentation.
*   `scripts/`: Management and migration scripts.

## Agent API

IntentKit provides a powerful REST API for programmatic control of your agents.

**Learn more:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

*   Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests.
*   Follow the [Skill Development Guide](docs/contributing/skills.md) for instructions on creating new skills.

### Developer Community

Join our [Discord](https://discord.com/invite/crestal) and apply for the "intentkit dev" role to collaborate with other developers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.