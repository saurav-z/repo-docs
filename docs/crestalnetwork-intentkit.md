# IntentKit: Build and Manage Intelligent AI Agents

**Unlock the power of autonomous AI agents with IntentKit, a versatile framework for blockchain interaction, social media management, and more.** ([View on GitHub](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to create and manage sophisticated AI agents capable of a wide range of tasks. This open-source framework streamlines the development process, allowing you to focus on building intelligent solutions.

## Key Features

*   **🤖 Multi-Agent Support:** Manage and deploy multiple autonomous agents.
*   **🔄 Autonomous Agent Management:** Full lifecycle management of your AI agents.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains (with more chains to come).
*   **🐦 Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:** Easily integrate custom skills and functionalities.
*   **🔌 MCP (WIP):** (Mention of MCP but no details, consider removing or providing more information)

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility:

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

For a more detailed explanation, consult the [Architecture](docs/architecture.md) documentation.

## Getting Started

### Package Manager Migration Warning

**Important:** We've switched to `uv` from `poetry` for package management. To set up your environment:

```bash
rm -rf .venv
uv sync
```

## Development

*   **Setup:** Refer to the [Development Guide](DEVELOPMENT.md) for setup instructions.
*   **Documentation:** Explore the [Documentation](docs/) for comprehensive information.

## Project Structure

The project is structured into core components and applications:

*   **[intentkit/](intentkit/)**: The core IntentKit package.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System-level configurations.
    *   [core/](intentkit/core/): Core agent system.
    *   [models/](intentkit/models/): Entity models.
    *   [skills/](intentkit/skills/): Extensible skills system.
    *   [utils/](intentkit/utils/): Utility functions.

*   **[app/](app/)**: The IntentKit application (API server, runner, scheduler).
    *   [admin/](app/admin/): Admin APIs.
    *   [entrypoints/](app/entrypoints/): Agent interaction entrypoints.
    *   [services/](app/services/): Service implementations.
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking.
    *   [readonly.py](app/readonly.py): Readonly entrypoint.
    *   [scheduler.py](app/scheduler.py): Task scheduler.
    *   [singleton.py](app/singleton.py): Agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Scripts for management and migrations

## Agent API

Programmatically interact with your agents using the comprehensive REST API.

*   **Access the API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions!

*   **Guidelines:** Review the [Contributing Guidelines](CONTRIBUTING.md).
*   **Contribute Skills:** Check the [Wishlist](docs/contributing/wishlist.md) for current requests.
    *   Refer to the [Skill Development Guide](docs/contributing/skills.md) for guidance.
*   **Join the Community:** Join our [Discord](https://discord.com/invite/crestal) and apply for the `intentkit dev` role for discussions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
Key improvements and explanations:

*   **SEO-Optimized Title & Hook:** The title directly states the core functionality and includes keywords like "AI agents" and "autonomous." The one-sentence hook immediately introduces the value proposition.
*   **Clear Headings:**  Uses proper headings for readability and structure, improving SEO.
*   **Bulleted Key Features:**  Highlights the core functionalities in an easy-to-scan bulleted list. This is great for users and also improves SEO by emphasizing keywords.
*   **Concise Language:**  Rewrites sections to be more direct and easy to understand, without losing technical accuracy.
*   **Actionable Instructions:**  Provides clear instructions for getting started and managing the package manager migration.
*   **Detailed Project Structure:**  Keeps the project structure section, vital for new users, but organizes it better.
*   **Contribution & Community Links:**  Includes and highlights the contribution guidelines and community links, encouraging engagement.
*   **Clearer Architecture Explanation:**  Emphasizes the flexibility and extensibility of the design.
*   **Removed Redundancy:** Streamlined the text.
*   **GitHub Link:** Added the prominent "View on GitHub" link at the beginning, driving traffic back to the repository.