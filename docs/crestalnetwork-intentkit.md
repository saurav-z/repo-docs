# IntentKit: Build Autonomous AI Agents with Ease

**[IntentKit](https://github.com/crestalnetwork/intentkit) empowers developers to create and manage intelligent, autonomous AI agents capable of interacting with blockchains, social media, and more.**

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features of IntentKit

*   **🤖 Multi-Agent Support:** Manage and deploy multiple AI agents.
*   **🔄 Autonomous Agent Management:** Control and orchestrate agent workflows.
*   **🔗 Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains.
*   **🐦 Social Media Integration:** Connect with Twitter, Telegram, and other platforms.
*   **🛠️ Extensible Skill System:**  Add custom capabilities using LangChain tools.
*   **🔌 MCP (WIP):** (Mentioned as "in progress")

## Architecture Overview

IntentKit utilizes a robust architecture leveraging LangGraph to power its AI agents. This allows for modularity and extensibility, enabling developers to easily integrate new functionalities and skills.

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

For a more in-depth understanding, please refer to the [Architecture](docs/architecture.md) documentation.

## Project Structure

The project is structured into core packages and an application layer:

**IntentKit Package:**

*   `intentkit/`: Core package (published as a pip package)
    *   `abstracts/`: Abstract classes and interfaces
    *   `clients/`: Clients for external services
    *   `config/`: System level configurations
    *   `core/`: Core agent system (LangGraph driven)
    *   `models/`: Entity models (Pydantic and SQLAlchemy)
    *   `skills/`: Extensible skills system (LangChain tools)
    *   `utils/`: Utility functions

**IntentKit Application:**

*   `app/`: The IntentKit app (API server, autonomous runner, and background scheduler)
    *   `admin/`: Admin APIs, agent generators
    *   `entrypoints/`: Agent interaction entrypoints (web, Telegram, Twitter, etc.)
    *   `services/`: Service implementations (Telegram, Twitter, etc.)
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

IntentKit offers a comprehensive REST API for programmatic interaction with your AI agents. This API facilitates the building of custom applications, integration with existing systems, and the creation of personalized interfaces.

**Access the API Documentation:** [Agent API Documentation](docs/agent_api.md)

## Development

Get started with development by reviewing the [Development Guide](DEVELOPMENT.md).

## Documentation

Explore the complete documentation before you start [Documentation](docs/).

## Contributing

We welcome contributions!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contributing Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for active requests.
2.  Consult the [Skill Development Guide](docs/contributing/skills.md) for detailed instructions.

### Community and Support

Join the developer community on [Discord](https://discord.com/invite/crestal) and request an intentkit dev role. You can also find a dedicated discussion channel there.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Package Manager Migration Warning

We have just migrated to `uv` from poetry.

To resolve the package manager migration, you need to delete the `.venv` folder and run `uv sync`. (one time)

```bash
rm -rf .venv
uv sync
```