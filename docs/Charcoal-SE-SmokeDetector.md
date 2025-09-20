# SmokeDetector: Your Headless Chatbot for Real-Time Spam Detection

Tired of spam cluttering your Stack Exchange chatrooms? SmokeDetector is a powerful, open-source chatbot designed to automatically detect and report spam in real-time. Learn more and contribute at the [original repository](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector leverages the power of:

*   **Real-Time Detection:** Monitors the Stack Exchange realtime tab for new questions.
*   **Automated Reporting:** Posts detected spam to designated chatrooms.
*   **Stack Exchange Integration:** Uses the Stack Exchange API and ChatExchange library for seamless interaction.

## Key Features

*   **Automated Spam Detection:** Identifies and reports spam based on predefined rules and patterns.
*   **Customizable Configuration:** Allows users to configure detection parameters, chatroom settings, and more.
*   **Flexible Deployment Options:** Supports setup through git, virtual environments, and Docker containers.
*   **Community-Driven:** Open-source project with active community contributions and improvements.

## Getting Started

### Installation

Follow these steps to get SmokeDetector up and running:

**Using Git, pip, and Python**

1.  Clone the repository:
    ```shell
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git checkout deploy
    ```
2.  Install dependencies:
    ```shell
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
3.  Configure SmokeDetector:
    *   Copy `config.sample` to a new file named `config`.
    *   Edit the values in the `config` file with your specific settings.
4.  Run SmokeDetector:
    ```shell
    python3 nocrash.py
    ```
    *   Consider running in a daemon-able mode like `screen`.

**Virtual Environment Setup**

1.  Clone and configure git environment:
    ```shell
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git config user.email "smokey@erwaysoftware.com"
    git config user.name "SmokeDetector"
    git checkout deploy
    ```
2.  Create a virtual environment:
    ```shell
    python3 -m venv env
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
3.  Configure SmokeDetector:
    *   Copy `config.sample` to a new file named `config`.
    *   Edit the values in the `config` file with your specific settings.
4.  Run SmokeDetector:
    ```shell
    env/bin/python3 nocrash.py
    ```

**Docker Setup**

1.  Build the Docker image:
    ```shell
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  Create a Docker container:
    ```shell
    docker create --name=mysmokedetector smokey:$DATE
    ```
3.  Configure SmokeDetector in container:
    *   Copy `config.sample` to a new file named `config`.
    *   Edit the values in the `config` file with your specific settings.
    ```shell
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```
4.  If you want to setup additional configurations within the container:
    ```shell
    docker exec -it mysmokedetector bash
    ```
5.  Ready the container for start:
    ```shell
    touch ~smokey/ready
    ```

**Docker Compose Setup**

1.  Prepare your `config` file based on `config.sample`.
2.  Create a directory, place your `config` file and `docker-compose.yml` file.
3.  Run `docker-compose up -d`

**Docker Compose Advanced Configuration**

Customize with:

```yaml
restart: always  # when your host reboots Smokey can autostart
mem_limit: 512M
cpus: 0.5  # Recommend 2.0 or more for spam waves
```

## Requirements

*   **Python:** Supports Python versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/).
*   **Stack Exchange Login:**  SmokeDetector requires Stack Exchange logins.
*   **Git (Recommended):**  Git 1.8 or higher (2.11+ recommended) for blacklist and watchlist modifications.

## Blacklist Removal Process

For requests to remove a website from the blacklist, please see the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" documentation.

## License

SmokeDetector is licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) or the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>) at your option.

### Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.