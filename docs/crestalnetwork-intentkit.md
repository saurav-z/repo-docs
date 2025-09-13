# IntentKit: Build Autonomous AI Agents with Ease

**Unleash the power of AI agents for blockchain interaction, social media management, and beyond with IntentKit, a cutting-edge autonomous agent framework.**  ([View the original repo](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to create and manage sophisticated AI agents capable of various tasks, from interacting with EVM blockchains to managing social media accounts.  Built with LangGraph, it offers a flexible and extensible architecture for building intelligent and autonomous systems.

## Key Features

*   🤖 **Multiple Agent Support:** Manage and orchestrate multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:** Enables agents to operate with minimal human intervention.
*   🔗 **Blockchain Integration:**  Seamlessly interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect and manage agents across various social media platforms (Twitter, Telegram, and more).
*   🛠️ **Extensible Skill System:**  Integrate custom skills to expand agent capabilities.
*   🔌 **MCP (WIP):**  Multi-Chain Provider support is in development.

## Architecture Overview

IntentKit's architecture is designed for modularity and extensibility.  Agents interact with various entrypoints (e.g., Twitter, Telegram) and leverage a core system powered by LangGraph. This system manages agent configuration, memory, and skills.  Skills, which can include blockchain interaction, internet search, and image processing, extend agent capabilities.

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

For a more in-depth understanding, refer to the detailed [Architecture](docs/architecture.md) documentation.

## Quick Start

### Package Manager Migration Warning

A package manager migration to `uv` from `poetry` has occurred.  To update your environment, please run the following commands:

```bash
rm -rf .venv
uv sync
```

## Development

*   **Development Guide:**  Get started with your setup by reading the [Development Guide](DEVELOPMENT.md).
*   **Documentation:** Explore the comprehensive [Documentation](docs/) for in-depth information.

## Project Structure

IntentKit is organized into a core package and an application:

*   **`intentkit/`**: The core IntentKit package (published as a pip package)
    *   `abstracts/`: Abstract classes and interfaces
    *   `clients/`: Clients for external services
    *   `config/`: System-level configurations
    *   `core/`: Core agent system, driven by LangGraph
    *   `models/`: Entity models using Pydantic and SQLAlchemy
    *   `skills/`: Extensible skills system, based on LangChain tools
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

IntentKit provides a robust REST API for seamless programmatic control of your AI agents.

*   **Agent API Documentation:** Learn how to build applications and integrate with existing systems using the [Agent API Documentation](docs/agent_api.md).

## Contribute

We welcome contributions!

*   **Contributing Guidelines:** Review the [Contributing Guidelines](CONTRIBUTING.md) before submitting your pull requests.
*   **Contribute Skills:**  Check the [Wishlist](docs/contributing/wishlist.md) for feature requests. The [Skill Development Guide](docs/contributing/skills.md) provides information for building custom skills.
*   **Developer Chat:**  Join the developer community on [Discord](https://discord.com/invite/crestal).  Apply for an IntentKit dev role after joining.  The discussions channel is open to contributors.

## License

IntentKit is licensed under the [MIT License](LICENSE).