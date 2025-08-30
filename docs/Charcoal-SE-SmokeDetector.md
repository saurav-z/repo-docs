# SmokeDetector: Your Headless Spam Hunter for Stack Exchange

**Protect your Stack Exchange community from spam with SmokeDetector, a powerful, headless chatbot that swiftly identifies and reports malicious content.**  For more details and to contribute, visit the [original repository](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

## Key Features

*   **Real-time Spam Detection:** Monitors Stack Exchange's realtime feed for potentially malicious content.
*   **Automated Reporting:** Posts spam reports to designated chatrooms.
*   **Flexible Configuration:** Customizable through a `config` file.
*   **Multiple Setup Options:** Supports installation via standard setup, virtual environments, and Docker.
*   **Open Source:**  Available under Apache 2.0 and MIT licenses.

## How SmokeDetector Works

SmokeDetector uses the [ChatExchange](https://github.com/Manishearth/ChatExchange) library to connect to chatrooms, gathers questions from the Stack Exchange [realtime tab](https://stackexchange.com/questions?tab=realtime), and accesses answers via the [Stack Exchange API](https://api.stackexchange.com/) to identify and flag potential spam.

Here's an example of how a chat post looks:

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Installation and Setup

Detailed setup instructions are available in the [wiki](https://charcoal-se.org/smokey).

### Basic Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Checkout the deployment branch `git checkout deploy`
4.  Install dependencies: `sudo pip3 install -r requirements.txt --upgrade` and `pip3 install --user -r user_requirements.txt --upgrade`
5.  Copy the sample configuration: Copy `config.sample` to `config`
6.  Edit the configuration file: Modify the values in `config` with your specific settings.
7.  Run SmokeDetector: `python3 nocrash.py` (recommended) or `python3 ws.py`

### Virtual Environment Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Configure git settings:
    `git config user.email "smokey@erwaysoftware.com"`
    `git config user.name "SmokeDetector"`
4.  Checkout the deployment branch `git checkout deploy`
5.  Create a virtual environment: `python3 -m venv env`
6.  Install dependencies: `env/bin/pip3 install -r requirements.txt --upgrade` and `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
7.  Copy the sample configuration: Copy `config.sample` to `config`
8.  Edit the configuration file: Modify the values in `config` with your specific settings.
9.  Run SmokeDetector: `env/bin/python3 nocrash.py`

### Docker Setup

1.  **Build the Docker image:**
    ```bash
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  **Create a container:**
    ```bash
    docker create --name=mysmokedetector smokey:$DATE
    ```
3.  **Configure:** Copy `config.sample` to a new file named `config`, edit the values required, then copy the file into the container:
    ```bash
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```
4.  **(Optional) Access the container:** `docker exec -it mysmokedetector bash` to configure further settings
5.  **(Optional) Ready:** In the container put a file named `ready` under `/home/smokey`: `touch ~smokey/ready`

#### Automate Docker deployment with Docker Compose

You can use `docker-compose` to automate the deployment of SmokeDetector.
You'll need a `config` file and a `docker-compose.yml` file.
Run `docker-compose up -d`.
You can configure resource limits in `docker-compose.yml`.

## Requirements

*   **Python:** SmokeDetector supports Python versions that are actively maintained.
*   **Git:** Git 1.8 or higher (2.11+ recommended) is required for blacklist/watchlist modifications.
*   **Stack Exchange Login:** SmokeDetector requires a valid Stack Exchange login.

## Blacklist Removal Process

For information on requesting removal of a website from the blacklist, see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)".

## License

SmokeDetector is licensed under the following:

*   Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)