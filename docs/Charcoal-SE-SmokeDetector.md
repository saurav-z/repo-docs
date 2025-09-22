# SmokeDetector: Real-Time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, open-source chatbot designed to identify and report spam on the Stack Exchange network in real-time.** This project actively monitors the platform and posts spam detections to designated chatrooms, helping to keep the community clean.

[Check out the original repo](https://github.com/Charcoal-SE/SmokeDetector)

## Key Features:

*   **Real-time Spam Detection:** Monitors Stack Exchange questions in real-time for spam.
*   **Automated Chat Reporting:** Posts spam detections to designated chatrooms for community review.
*   **Headless Chatbot:** Operates as a background process, integrating seamlessly with chat platforms.
*   **Uses ChatExchange:** Leverages the ChatExchange library for efficient chat interaction.
*   **API Integration:** Accesses question data via the Stack Exchange API for accurate analysis.

## Getting Started:

Detailed setup instructions are available in the [wiki](https://charcoal-se.org/smokey).

### Installation Options:

SmokeDetector offers multiple installation methods to suit your needs:

*   **Basic Setup:**
    1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
    2.  Navigate to the directory: `cd SmokeDetector`
    3.  Switch to the deploy branch: `git checkout deploy`
    4.  Install dependencies: `sudo pip3 install -r requirements.txt --upgrade` and `pip3 install --user -r user_requirements.txt --upgrade`
    5.  Configure SmokeDetector: Copy `config.sample` to `config` and edit the required values.
    6.  Run SmokeDetector: `python3 nocrash.py` (recommended for continuous operation) or `python3 ws.py` (limited 6-hour run time).
*   **Virtual Environment Setup:** Uses a virtual environment for dependency isolation. Steps similar to basic setup, with the addition of creating and activating a Python virtual environment: `python3 -m venv env` and `env/bin/pip3 install ...`
*   **Docker Setup:** Uses Docker containers for advanced isolation. Includes instructions for building a Docker image, creating a container, and running the application. (See original README for detailed Docker setup instructions.)
*   **Docker Compose Setup:** Provides instructions for using Docker Compose for automated deployment, including configuration options for resource management.

## Requirements:

*   **Stack Exchange Login:** Requires valid Stack Exchange login credentials.
*   **Python:** Supports Python versions within the [supported phase of the Python life cycle](https://devguide.python.org/versions/)
*   **Git (for blacklist/watchlist updates):** Git 1.8 or higher (2.11+ recommended)

## Blacklist Removal Requests

For official representatives of websites/products, please refer to the [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal) to request removal from the blacklist.

## License

SmokeDetector is available under the following licenses:

*   Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

## Contribution Licensing

Contributions are dual-licensed under the terms of both the Apache-2.0 and MIT licenses.