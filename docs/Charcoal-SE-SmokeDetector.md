# SmokeDetector: Real-time Spam Detection for Stack Exchange

Tired of spam cluttering your Stack Exchange chats? SmokeDetector, a headless chatbot, efficiently detects and reports spam in real-time. ([View on GitHub](https://github.com/Charcoal-SE/SmokeDetector))

**Key Features:**

*   **Real-time Spam Detection:** Monitors Stack Exchange's realtime tab for suspicious activity.
*   **Automated Reporting:** Posts detected spam to designated chatrooms for review and action.
*   **Uses Stack Exchange API:** Leverages the official API for accessing questions and answers.
*   **Flexible Setup:** Supports various setup methods, including virtual environments and Docker containers.
*   **Customizable:** Allows users to configure settings to tailor spam detection behavior.

**Installation & Setup:**

SmokeDetector offers various installation options, including:

*   **Basic Setup:** Simple instructions to get SmokeDetector running quickly.
*   **Virtual Environment Setup:** Recommended for isolating dependencies and maintaining a clean environment.
*   **Docker Setup:** Provides containerization for easy deployment and management.
*   **Docker Compose Setup:** Automates deployment with configuration via `docker-compose.yml` files.

Detailed setup instructions, including requirements and configuration guides, can be found in the project's [wiki](https://charcoal-se.org/smokey).

**Requirements:**

*   Python (Supported versions as defined in the [Python life cycle](https://devguide.python.org/versions/))
*   Stack Exchange Login
*   Git 1.8 or higher (2.11+ recommended) for blacklist/watchlist modifications.

**License:**

SmokeDetector is licensed under either the Apache License, Version 2.0, or the MIT license, at your option. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.