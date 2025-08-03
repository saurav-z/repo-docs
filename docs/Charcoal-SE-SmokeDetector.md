# SmokeDetector: Real-Time Spam Detection for Stack Exchange

Tired of spam cluttering your Stack Exchange communities? **SmokeDetector is a powerful, open-source chatbot that automatically identifies and reports spam in real-time.** This project is available on [GitHub](https://github.com/Charcoal-SE/SmokeDetector).

## Key Features:

*   **Automated Spam Detection:** Monitors the Stack Exchange real-time feed for suspicious content.
*   **Real-time Reporting:** Posts detected spam to designated chatrooms for moderator review and action.
*   **Headless Chatbot:** Operates seamlessly in the background, requiring minimal user interaction.
*   **Uses ChatExchange:** Leverages the ChatExchange library for efficient communication with chat rooms.
*   **Open Source:**  Contribute to the project and customize it to meet your community's needs.

## Getting Started

SmokeDetector is designed to be flexible and can be deployed using several methods:

*   **Basic Setup:** Follow the command-line instructions in the wiki to get the code, install requirements and configure your connection to Stack Exchange.
*   **Virtual Environment Setup:** Protect your system dependencies and ensure isolation using virtual environments.
*   **Docker Setup:**  Utilize Docker containers for a streamlined and portable deployment experience. Instructions for building an image and running a container are included in the main README.
*   **Docker Compose Setup:** Automate deployments with Docker Compose and add constraints like memory and CPU to manage resources.

Detailed setup and configuration instructions are available in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

## Requirements

*   **Python:** Supports Python versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/) (between "First release" and "End of life").
*   **Stack Exchange Login:** Requires a valid Stack Exchange account.
*   **Git (Recommended):** Git 1.8 or higher (2.11+ recommended) is required for committing blacklist/watchlist modifications.

## Blacklist Removal

If you are a representative of a website that you would like to have removed from the blacklist, please consult the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for instructions.

## License

SmokeDetector is licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) and the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>), at your option.