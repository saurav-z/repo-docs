# IntentKit: Build Autonomous AI Agents with Ease

**IntentKit empowers you to build and manage sophisticated AI agents capable of blockchain interaction, social media management, and custom skill execution, all within a unified framework.**  Explore the power of autonomous agents with IntentKit! ([Original Repository](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   🤖 **Multi-Agent Support:** Manage multiple AI agents, each with unique capabilities.
*   🔄 **Autonomous Agent Management:** Automate agent actions and workflows.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Customize agents with a flexible skill architecture.
*   🔌 **MCP (WIP):**  Modular Component for a variety of uses.

## Architecture Overview

IntentKit's architecture is designed for modularity and extensibility, with LangGraph at its core.

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

For a detailed understanding, refer to the [Architecture](docs/architecture.md) documentation.

## Quick Start:  Environment Setup

**Important:**  This project now uses `uv` for package management. If you are upgrading from a previous version, you *must* remove your existing virtual environment and re-install dependencies.

```bash
rm -rf .venv
uv sync
```

## Project Structure

IntentKit is organized into core components and an application layer:

*   **`intentkit/`**: The core IntentKit package (published as a pip package).  Includes abstracts, clients, config, core agent system, models, skills, and utility functions.
*   **`app/`**: The IntentKit application layer (API server, agent runner, and scheduler).  Includes admin, entrypoints, services, REST API, autonomous agent runner, checker, readonly, scheduler, singleton, Telegram and Twitter integrations.
*   `docs/`:  Documentation.
*   `scripts/`:  Management and migration scripts.

## Agent API

Access and control your agents programmatically through the IntentKit REST API.

*   **Explore the API:** [Agent API Documentation](docs/agent_api.md)

## Development

*   **Get Started:** [Development Guide](DEVELOPMENT.md)
*   **Explore the Docs:** [Documentation](docs/)

## Contributing

We welcome contributions!

*   **Contribution Guidelines:** [Contributing Guidelines](CONTRIBUTING.md)
*   **Contribute Skills:** Check the [Wishlist](docs/contributing/wishlist.md) and read the [Skill Development Guide](docs/contributing/skills.md).

### Community & Support

*   **Developer Chat:** Join us on [Discord](https://discord.com/invite/crestal).  Apply for a dev role for access to dedicated channels.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.