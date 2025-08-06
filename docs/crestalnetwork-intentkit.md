# IntentKit: Build and Manage Autonomous AI Agents

IntentKit empowers you to create and manage intelligent AI agents with diverse capabilities, from blockchain interactions to social media management.  [Learn more at the original repository](https://github.com/crestalnetwork/intentkit).

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features of IntentKit

*   🤖 **Multi-Agent Support:** Easily create and manage multiple autonomous agents.
*   🔄 **Autonomous Agent Management:** Simplify the lifecycle of your AI agents.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains for various on-chain actions.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram to interact with your audience.
*   🛠️ **Extensible Skill System:** Add custom skills to extend agent capabilities, with a base of LangChain tools.
*   🔌 **MCP (WIP):** Placeholder for upcoming features.

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility.  The agent system uses LangGraph at its core.

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

For a more detailed view, consult the [Architecture](docs/architecture.md) documentation.

## Development & Getting Started

### Package Manager Migration Warning

If you are setting up your environment, you need to remove the .venv folder and run `uv sync` to create a new virtual environment. (one time)

```bash
rm -rf .venv
uv sync
```

### Development Guide

To start developing with IntentKit, consult the [Development Guide](DEVELOPMENT.md).

### Documentation

Comprehensive documentation is available to guide you.  Explore the [Documentation](docs/) for detailed information.

## Project Structure

The project is organized into two main parts: the core package and the application.

*   **`intentkit/`**: The IntentKit package (published as a pip package)
    *   `abstracts/`: Abstract classes and interfaces
    *   `clients/`: Clients for external services
    *   `config/`: System level configurations
    *   `core/`: Core agent system, driven by LangGraph
    *   `models/`: Entity models using Pydantic and SQLAlchemy
    *   `skills/`: Extensible skills system
    *   `utils/`: Utility functions
*   **`app/`**: The IntentKit application (API server, autonomous runner, and background scheduler)
    *   `admin/`: Admin APIs, agent generators, and related functionality
    *   `entrypoints/`: Entrypoints for interacting with agents (web, Telegram, Twitter, etc.)
    *   `services/`: Service implementations for Telegram, Twitter, etc.
    *   `api.py`: REST API server
    *   `autonomous.py`: Autonomous agent runner
    *   `checker.py`: Health and credit checking logic
    *   `readonly.py`: Readonly entrypoint
    *   `scheduler.py`: Background task scheduler
    *   `singleton.py`: Singleton agent manager
    *   `telegram.py`: Telegram integration
    *   `twitter.py`: Twitter integration
*   `docs/`: Documentation
*   `scripts/`: Operation and temporary scripts

## Agent API

IntentKit offers a robust REST API for programmatic access to your agents. Use the Agent API to integrate IntentKit into existing systems, build custom interfaces, and build applications.

**Access the API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests.
2.  Follow the [Skill Development Guide](docs/contributing/skills.md) to contribute a skill.

### Developer Chat

Join our [Discord](https://discord.com/invite/crestal) to connect with the IntentKit community and apply for a dev role.

## License

IntentKit is licensed under the [MIT License](LICENSE).