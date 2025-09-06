# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Tired of spam polluting your Stack Exchange communities?** SmokeDetector is a powerful, headless chatbot designed to automatically identify and report spam in real-time, keeping your platforms clean and your users engaged. [Visit the original repository for more details.](https://github.com/Charcoal-SE/SmokeDetector)

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector leverages the Stack Exchange API and ChatExchange to monitor the real-time questions feed, identifying spam and posting alerts to designated chatrooms. See an example chat post below:

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Key Features:

*   **Real-time Spam Detection:** Continuously monitors Stack Exchange for spam.
*   **Automated Reporting:** Posts spam alerts to chatrooms.
*   **Open Source:**  Contribute and customize to meet your needs.
*   **Flexible Setup:** Supports various setup options including basic, virtual environment, and Docker.

## Documentation

Comprehensive user documentation is available in the [wiki](https://charcoal-se.org/smokey). Detailed setup and running guides are also available.

## Setup & Installation

### Basic Setup

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
3.  Configure: Copy `config.sample` to `config` and edit the required values.
4.  Run:  `python3 nocrash.py` (recommended for continuous operation) or `python3 ws.py`.

### Virtual Environment Setup

1.  Follow steps 1-3 from the Basic Setup.
2.  Create a virtual environment:
    ```shell
    python3 -m venv env
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
3.  Configure the `config` file, then run SmokeDetector using `env/bin/python3 nocrash.py`.

### Docker Setup

1.  Build the Docker image:

    ```shell
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  Create a container:
    ```shell
    docker create --name=mysmokedetector smokey:$DATE
    ```
3.  Configure and copy the config file into the container:
    ```shell
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```
4.  Start the container: `docker start mysmokedetector`
5.  Customize (optional): `docker exec -it mysmokedetector bash`

#### Automate Docker Deployment with Docker Compose

1.  Ensure you have a configured `config` file.
2.  Create a directory, place `config` and [`docker-compose.yml`](docker-compose.yml) in it.
3.  Run: `docker-compose up -d`

### Additional Docker Compose Configuration
For more control over resources and auto-restart behavior, modify the `docker-compose.yml` file and add the following parameters to the `smokey` section:

```yaml
restart: always  # when your host reboots Smokey can autostart
mem_limit: 512M
cpus: 0.5  # Recommend 2.0 or more for spam waves
```
## Requirements

SmokeDetector requires Python versions that are within the supported phase of the Python life cycle and supports Stack Exchange logins. Git 1.8+ (2.11+ recommended) is also needed for blacklist/watchlist modifications.

## Blacklist Removal

For official representatives seeking website removal from the blacklist, please refer to the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for instructions.

## License

This project is dual-licensed under the Apache License 2.0 and the MIT license. See [LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0> and [LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT> for more information.