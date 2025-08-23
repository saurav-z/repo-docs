# IntentKit: Build Autonomous AI Agents for a Smarter World

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

**IntentKit empowers you to build, manage, and deploy intelligent AI agents with diverse capabilities, enabling automation across blockchain, social media, and custom integrations.**  Dive into the future of AI agents with IntentKit, and explore the [original repository](https://github.com/crestalnetwork/intentkit) for the latest updates and to contribute.

## Key Features

*   **🤖 Multi-Agent Support:** Manage and deploy multiple autonomous AI agents.
*   **🔄 Autonomous Agent Management:**  Automate agent lifecycle and operations.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains for on-chain actions.
*   **🐦 Social Media Integration:** Seamlessly manage content and interactions on platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:**  Easily integrate new skills to expand agent capabilities, built on LangChain tools.
*   **🔌 MCP (Work in Progress):** Explore the future of the IntentKit framework.

## Architecture

IntentKit's architecture is designed for flexibility and extensibility.  It leverages LangGraph to power the core agent system, connecting to various entrypoints and skills.

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

For a deeper dive into the architecture, refer to the [Architecture](docs/architecture.md) section in the documentation.

## Development

Get started with your IntentKit development environment by reading the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to guide you. Explore the [Documentation](docs/) to learn more about IntentKit's features and functionality.

## Project Structure

IntentKit is organized into two main parts: the core package and the application.

**IntentKit Package (`intentkit/`)**

*   `abstracts/`: Abstract classes and interfaces
*   `clients/`: Clients for external services
*   `config/`: System-level configurations
*   `core/`: Core agent system
*   `models/`: Entity models
*   `skills/`: Extensible skills system
*   `utils/`: Utility functions

**IntentKit Application (`app/`)**

*   `admin/`: Admin APIs and agent generators
*   `entrypoints/`: Agent interaction entrypoints (web, Telegram, Twitter, etc.)
*   `services/`: Service implementations
*   `api.py`: REST API server
*   `autonomous.py`: Autonomous agent runner
*   `checker.py`: Health and credit checking logic
*   `readonly.py`: Read-only entrypoint
*   `scheduler.py`: Background task scheduler
*   `singleton.py`: Singleton agent manager
*   `telegram.py`: Telegram integration
*   `twitter.py`: Twitter integration

Other important directories:

*   `docs/`: Documentation
*   `scripts/`: Management and migration scripts

## Agent API

Programmatically access your agents and integrate them with other systems using the comprehensive REST API.

**Explore the API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

Contribute new skills to expand the capabilities of IntentKit agents. Check the [Wishlist](docs/contributing/wishlist.md) for current requests.

To get started with skill development, refer to the [Skill Development Guide](docs/contributing/skills.md).

### Developer Community

Join the IntentKit developer community on [Discord](https://discord.com/invite/crestal).  Open a support ticket to apply for an intentkit dev role and connect with other developers.

## Package Manager Migration Notice

To update your development environment, follow these steps:
1.  Delete your existing virtual environment using `rm -rf .venv`
2.  Create a new environment using `uv sync`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.