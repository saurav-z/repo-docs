# IntentKit: Build and Manage Powerful AI Agents

**Unlock the power of autonomous AI agents for blockchain, social media, and custom tasks with IntentKit, a robust and extensible framework.**  ([View the original repository](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers you to create and manage AI agents capable of complex interactions, from blockchain transactions to social media engagement.  This framework offers a modular design with a strong emphasis on extensibility, allowing you to tailor your agents to specific needs.

## Key Features

*   🤖 **Multiple Agent Support:** Manage and orchestrate multiple independent AI agents.
*   🔄 **Autonomous Agent Management:**  Ensure agents operate and evolve without direct human input.
*   🔗 **Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:**  Connect with platforms like Twitter, Telegram, and more.
*   🛠️ **Extensible Skill System:**  Customize agent capabilities with a flexible skill architecture.
*   🔌 **MCP (WIP):** (Further details to be added when feature is ready)

## Architecture

IntentKit's architecture centers around a core agent system powered by LangGraph, enabling sophisticated interactions and decision-making. This simplified view highlights key components:

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

For a comprehensive understanding of the architecture, refer to the detailed [Architecture documentation](docs/architecture.md).

## Getting Started

### Package Manager Migration Notice

We have migrated to `uv` from `poetry`.  To set up your environment:

```bash
rm -rf .venv  # Delete the old virtual environment
uv sync        # Create a new virtual environment
```

## Development

*   **Development Guide:** Dive into the development process with our [Development Guide](DEVELOPMENT.md).
*   **Documentation:** Consult the comprehensive [Documentation](docs/) for in-depth information.

## Project Structure

The IntentKit project is organized into two primary components: the core package and the application.

*   **`intentkit/` (Core Package):**  The pip-installable package containing the core framework.
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for interacting with external services.
    *   `config/`: System-level configurations.
    *   `core/`:  The core agent system.
    *   `models/`: Data models using Pydantic and SQLAlchemy.
    *   `skills/`: The extensible skill system.
    *   `utils/`: Utility functions.

*   **`app/` (Application):** The application layer, including the API server, autonomous runner, and scheduler.
    *   `admin/`: Admin APIs.
    *   `entrypoints/`: Entrypoints (web, Telegram, Twitter, etc.).
    *   `services/`: Service implementations.
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking logic.
    *   `readonly.py`: Readonly entrypoint
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.

*   `docs/`: Documentation.
*   `scripts/`:  Management and migration scripts.

## Agent API

**Integrate seamlessly with your agents using our REST API.**

*   **Get Started:** Explore the [Agent API Documentation](docs/agent_api.md) for full access details.

## Contributing

We welcome contributions!  Please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contributing Skills

*   **Wishlist:** Check the [Wishlist](docs/contributing/wishlist.md) for current skill requests.
*   **Skill Development Guide:**  Follow the [Skill Development Guide](docs/contributing/skills.md) to create new skills.

### Developer Community

*   **Discord:** Join our vibrant developer community on [Discord](https://discord.com/invite/crestal) and request an IntentKit dev role.
*   **Discussion:** Engage with developers in our dedicated discussion channel.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for more information.