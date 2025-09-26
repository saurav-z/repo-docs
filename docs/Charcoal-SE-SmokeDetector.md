# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Tired of spam on Stack Exchange? SmokeDetector, a headless chatbot, tirelessly identifies and reports malicious content, keeping your community clean.**

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector is a powerful, headless chatbot specifically designed to combat spam on Stack Exchange. It actively monitors the real-time feed and utilizes the Stack Exchange API to quickly identify and report harmful content to designated chatrooms. Check out an example [chat post](https://chat.stackexchange.com/transcript/message/43579469):

![Example chat post](https://i.sstatic.net/oLyfb.png)

**Key Features:**

*   **Real-time Spam Detection:** Continuously monitors Stack Exchange for suspicious activity.
*   **Automated Reporting:** Posts spam reports to designated chatrooms, alerting moderators and users.
*   **ChatExchange Integration:** Leverages ChatExchange for seamless communication and reporting.
*   **API-Driven:** Uses the Stack Exchange API to efficiently access and analyze content.
*   **Flexible Deployment:** Supports multiple setup options, including direct install, virtual environments, and Docker containers.

## Getting Started

For detailed setup instructions and user documentation, please refer to the [SmokeDetector Wiki](https://charcoal-se.org/smokey/).

### Basic Setup

1.  **Clone the repository:** `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  **Navigate to the directory:** `cd SmokeDetector`
3.  **Checkout the deploy branch:** `git checkout deploy`
4.  **Install dependencies:**
    ```bash
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
5.  **Configure SmokeDetector:** Copy `config.sample` to `config` and edit with your specific settings.
6.  **Run SmokeDetector:** Execute `python3 nocrash.py` (recommended for automatic restarts) or `python3 ws.py` (shuts down after 6 hours).

### Virtual Environment Setup

Using a virtual environment is recommended for dependency isolation.

1.  Follow steps 1-3 from Basic Setup.
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv env
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
3.  **Configure SmokeDetector:** Copy `config.sample` to `config` and edit.
4.  **Run SmokeDetector:** Execute `env/bin/python3 nocrash.py`.

### Docker Setup

Docker provides the best isolation for dependencies.

1.  **Build the Docker image:**
    ```bash
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  **Create a Docker container:** `docker create --name=mysmokedetector smokey:$DATE`
3.  **Configure SmokeDetector:** Copy `config.sample` to `config` in your local directory, edit the values, then copy it to the container: `docker cp config mysmokedetector:/home/smokey/SmokeDetector/config`
4.  **Start the container:** After configuration is done, create a file named `ready` in the container under `/home/smokey`
    ```bash
    docker exec -it mysmokedetector bash
    touch ~smokey/ready
    ```
    See the README for additional Docker Compose setup instructions.

## Requirements

*   **Stack Exchange Login:** SmokeDetector requires a Stack Exchange login to function.
*   **Python:** Supports Python versions within the [supported phase of the Python life cycle](https://devguide.python.org/versions/).
*   **Git (for blacklist modifications):** Git 1.8+ (2.11+ recommended) is required for committing blacklist and watchlist changes.

## Blacklist Removal

If you represent a website and wish to have it removed from the blacklist, please follow the [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal).

## License

SmokeDetector is licensed under the terms of the:

*   Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

## Contribution Licensing

By contributing to SmokeDetector, you agree that your contributions are dual-licensed under the Apache 2.0 and MIT licenses.

**[Back to the original repository](https://github.com/Charcoal-SE/SmokeDetector)**