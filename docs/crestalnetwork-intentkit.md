# IntentKit: Build Autonomous AI Agents with Ease

**Unlock the power of AI agents with IntentKit, a flexible framework for creating and managing intelligent, autonomous systems for various applications.**  (Check out the original repo: [https://github.com/crestalnetwork/intentkit](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is a powerful framework designed to simplify the development of AI agents. It offers a modular and extensible architecture, enabling you to build agents with diverse capabilities. Whether you're interacting with blockchains, managing social media, or integrating custom skills, IntentKit provides the tools you need.

## Key Features

*   🤖 **Multiple Agent Support:** Manage numerous AI agents concurrently.
*   🔄 **Autonomous Agent Management:** Built-in capabilities for autonomous operation and orchestration.
*   🔗 **Blockchain Integration:** Seamless interaction with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect and manage agents on platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Easily add new skills and functionalities to your agents.
*   🔌 **MCP (WIP):** (Mention the feature and its state)

## Architecture Overview

IntentKit's architecture is designed for flexibility and scalability. Agents are powered by LangGraph, and interact with various services and skills.

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

For a more in-depth understanding, refer to the [Architecture](docs/architecture.md) section.

## Development

Get started with your IntentKit setup: [Development Guide](DEVELOPMENT.md)

## Documentation & Resources

*   [Documentation](docs/)
*   [Agent API Documentation](docs/agent_api.md)

## Project Structure

The project is organized into the core package and the application:

**IntentKit Package:** (Published as a pip package -  `intentkit/`)

*   `abstracts/`: Abstract classes and interfaces
*   `clients/`: Clients for external services
*   `config/`: System level configurations
*   `core/`: Core agent system (LangGraph-driven)
*   `models/`: Entity models (Pydantic & SQLAlchemy)
*   `skills/`: Extensible skills system (based on LangChain tools)
*   `utils/`: Utility functions

**IntentKit App:** (API server, runner, and scheduler - `app/`)

*   `admin/`: Admin APIs
*   `entrypoints/`: Agent interaction entrypoints
*   `services/`: Service implementations (Telegram, Twitter, etc.)
*   `api.py`: REST API server
*   `autonomous.py`: Autonomous agent runner
*   `checker.py`: Health and credit checking logic
*   `readonly.py`: Readonly entrypoint
*   `scheduler.py`: Background task scheduler
*   `singleton.py`: Singleton agent manager
*   `telegram.py`: Telegram integration
*   `twitter.py`: Twitter integration

**Additional Directories:**

*   `docs/`: Documentation
*   `scripts/`: Management and migration scripts

## Agent API

IntentKit provides a comprehensive REST API for programmatic access to your agents. Build applications, integrate with existing systems, or create custom interfaces using our Agent API.

## Contributing

We welcome contributions!  Review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

Start by checking the [Wishlist](docs/contributing/wishlist.md) for active requests.

Then, follow the [Skill Development Guide](docs/contributing/skills.md).

### Developer Chat

Join the IntentKit community on [Discord](https://discord.com/invite/crestal) and apply for an intentkit dev role in a support ticket.

## License

IntentKit is released under the [MIT License](LICENSE).