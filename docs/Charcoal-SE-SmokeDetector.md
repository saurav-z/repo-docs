# SmokeDetector: Your AI-Powered Spam Hunter for Stack Exchange

SmokeDetector is a headless chatbot designed to combat spam on Stack Exchange, ensuring a cleaner and more reliable platform for everyone. For more information and to contribute, visit the original repository: [https://github.com/Charcoal-SE/SmokeDetector](https://github.com/Charcoal-SE/SmokeDetector)

## Key Features:

*   **Automated Spam Detection:** Utilizes advanced algorithms to identify and flag spam posts in real-time.
*   **Real-time Monitoring:** Continuously monitors the Stack Exchange realtime tab for new questions.
*   **Chat Integration:** Posts detected spam to designated chatrooms for review and action.
*   **Stack Exchange API Integration:** Leverages the Stack Exchange API to access question and answer data.
*   **Flexible Deployment:** Supports various setup options, including:
    *   **Standard Setup:** Quick and easy installation using pip.
    *   **Virtual Environment Setup:** Isolates dependencies for cleaner installations.
    *   **Docker Setup:** Containerizes SmokeDetector for portability and resource management.
    *   **Docker Compose:** Orchestrates SmokeDetector with memory, CPU, and auto-restart options.

## Getting Started

Follow the [detailed setup instructions](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector) in the wiki.

## Requirements

*   Python (See supported versions [here](https://devguide.python.org/versions/))
*   Stack Exchange Login
*   Git 1.8+ (2.11+ recommended) for blacklist and watchlist modifications.

## Blacklist Removal Process

Official representatives of websites or products can request removal from the blacklist. Please see the detailed instructions in the [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal).

## License

SmokeDetector is licensed under the following options:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

## Contribution Licensing

By contributing to SmokeDetector, you agree to the dual licensing terms as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).