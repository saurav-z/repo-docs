# SmokeDetector: Real-Time Spam Detection for Stack Exchange Chat

**Tired of spam cluttering your Stack Exchange communities? SmokeDetector is a headless chatbot designed to identify and report spam in real-time, keeping your chatrooms clean.** [Visit the original repo on GitHub](https://github.com/Charcoal-SE/SmokeDetector).

## Key Features:

*   **Real-time Spam Detection:** Monitors Stack Exchange's real-time question feed for suspicious content.
*   **Automated Reporting:** Posts detected spam to designated chatrooms, alerting moderators and users.
*   **ChatExchange Integration:** Utilizes the ChatExchange library for seamless interaction with chat platforms.
*   **API Access:** Leverages the Stack Exchange API to access and analyze question and answer data.
*   **Flexible Deployment:** Supports multiple deployment options including:
    *   **Basic Setup:** Simple instructions for quick setup using `git clone`, `pip3 install`, and configuration.
    *   **Virtual Environment Setup:** Isolates dependencies for a clean and manageable environment.
    *   **Docker Setup:** Utilizes Docker containers for easy and consistent deployment.
    *   **Docker Compose Setup:** Automates Docker deployment for ease of use.

## Setup and Configuration:

Detailed setup instructions are available in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector). The wiki also contains detailed instructions for [setting up and running SmokeDetector](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

## Requirements:

*   Stack Exchange Login
*   Python versions within the [supported phase of the Python life cycle](https://devguide.python.org/versions/)
*   Git 1.8 or higher (2.11+ recommended) for blacklist/watchlist modifications.

## Blacklist Removal:

For website removal requests, please refer to the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" in the wiki.

## License:

Licensed under either of:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

### Contribution Licensing:

By submitting your contribution for inclusion in the work as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0), you agree that it be dual licensed as above, without any additional terms or conditions.