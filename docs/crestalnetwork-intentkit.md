# IntentKit: Build Autonomous AI Agents with Ease

**Unleash the power of AI with IntentKit, an open-source framework for creating and managing intelligent autonomous agents.** (See the original repository on [GitHub](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to build AI agents with diverse capabilities, including blockchain interaction, social media management, and custom skill integration. This framework simplifies the complexities of agent development, allowing you to focus on innovation.

## Key Features

*   🤖 **Multi-Agent Support:** Manage and orchestrate multiple autonomous agents.
*   🔄 **Autonomous Agent Management:** Streamline the lifecycle of your AI agents.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter, Telegram, and more.
*   🛠️ **Extensible Skill System:** Customize agent behavior with a modular skill system.
*   🔌 **MCP (WIP):** (Further details to be added)

## Architecture Overview

IntentKit is designed with a flexible architecture. The core agent system is powered by LangGraph, enabling sophisticated control and logic.

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

For a more detailed view of the system, refer to the comprehensive [Architecture](docs/architecture.md) documentation.

## Development & Documentation

*   **Get Started:**  Follow the [Development Guide](DEVELOPMENT.md) to set up your development environment.
*   **Explore the Docs:** Consult the comprehensive [Documentation](docs/) for in-depth information and tutorials.

## Project Structure

The project is structured for clarity and maintainability, divided into two primary components:

*   **[intentkit/](intentkit/)**: The core IntentKit package (available on PyPI).
    *   Includes key modules like `abstracts`, `clients`, `config`, `core`, `models`, `skills`, and `utils`.
*   **[app/](app/)**: The IntentKit application, including:
    *   `admin`: Admin APIs and agent generation.
    *   `entrypoints`: Agent interaction entrypoints.
    *   `services`: Service implementations for integrations.
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking logic.
    *   `readonly.py`: Readonly entrypoint
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.
*   `docs/`: Documentation
*   `scripts/`: Management and migration scripts.

## Agent API

Programmatically interact with your agents through our robust REST API. Build custom applications and integrations with ease.

*   **API Documentation:** Access the [Agent API Documentation](docs/agent_api.md) to learn more.

## Contributing

We welcome contributions!

*   **Contribution Guidelines:** Review our [Contributing Guidelines](CONTRIBUTING.md) to get started.
*   **Skill Development:** Explore the [Wishlist](docs/contributing/wishlist.md) and the [Skill Development Guide](docs/contributing/skills.md) for skill contributions.
*   **Join the Community:** Connect with developers on our [Discord](https://discord.com/invite/crestal).

## License

IntentKit is licensed under the [MIT License](LICENSE).