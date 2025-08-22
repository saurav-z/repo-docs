# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Tired of spam cluttering your Stack Exchange chatrooms? SmokeDetector is your headless chatbot solution, automatically identifying and reporting spam in real-time.** ([Back to Original Repo](https://github.com/Charcoal-SE/SmokeDetector))

<p align="center">
  <img src="https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster" alt="Build Status">
  <img src="https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield" alt="Circle CI">
  <img src="https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master" alt="Coverage Status">
  <img src="https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg" alt="Open issues">
  <img src="https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg" alt="Open PRs">
</p>

SmokeDetector is a powerful headless chatbot that monitors Stack Exchange for spam, posting detections to chatrooms. It uses ChatExchange and the Stack Exchange API to quickly identify and report unwanted content.

Key Features:

*   **Real-time Spam Detection:** Identifies and reports spam as it appears on Stack Exchange.
*   **Automated Reporting:** Posts spam reports to designated chatrooms.
*   **Uses Stack Exchange API:** Leverages the official API for data access.
*   **Flexible Deployment:** Supports various setup methods, including virtual environments and Docker containers.
*   **Open Source:**  Available under Apache 2.0 and MIT licenses.

## Getting Started

### Setup and Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git checkout deploy
    ```
2.  **Install Dependencies:**  Choose a setup method:

    *   **Standard Setup:**
        ```bash
        sudo pip3 install -r requirements.txt --upgrade
        pip3 install --user -r user_requirements.txt --upgrade
        ```
    *   **Virtual Environment Setup:**  (Recommended for dependency isolation)
        ```bash
        python3 -m venv env
        env/bin/pip3 install -r requirements.txt --upgrade
        env/bin/pip3 install --user -r user_requirements.txt --upgrade
        ```
    *   **Docker Setup:** (Best for isolating dependencies) Follow the Docker setup instructions in the [original README](https://github.com/Charcoal-SE/SmokeDetector).
3.  **Configure:** Copy `config.sample` to `config` and edit with your specific settings (API keys, chat room IDs, etc.).
4.  **Run:**
    *   `python3 nocrash.py` (Recommended for continuous operation)
    *   `python3 ws.py` (Runs for a limited time, restarts not guaranteed)
    *   For virtual environments use: `env/bin/python3 nocrash.py`
    *   For Docker, follow the instructions in the original README to run the container.

## Documentation

Detailed documentation, including setup instructions and configuration guides, can be found in the [wiki](https://charcoal-se.org/smokey) (User documentation is in the [wiki](https://charcoal-se.org/smokey).).

## Requirements

*   **Python:** SmokeDetector supports Python versions which are in the [supported phase of the Python life cycle](https://devguide.python.org/versions/).
*   **Stack Exchange Login:** SmokeDetector requires Stack Exchange login credentials.
*   **Git:**  Git 1.8 or higher (Git 2.11+ recommended) for blacklist/watchlist modifications.

## Blacklist Removal

For website removal requests, please see the [process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal).

## License

SmokeDetector is available under the following licenses:

*   [Apache License, Version 2.0](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>
*   [MIT license](LICENSE-MIT) or <https://opensource.org/licenses/MIT>

## Contributing

By submitting your contribution, you agree that it will be dual licensed as above, without any additional terms or conditions.