# SmokeDetector: Your Headless Chatbot for Real-Time Spam Detection

**SmokeDetector is a powerful, headless chatbot that actively detects and flags spam on Stack Exchange, keeping your community clean and informed.** For more details, visit the original repository: [Charcoal-SE/SmokeDetector](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector utilizes the Stack Exchange API and ChatExchange to monitor the realtime tab and post detections to chatrooms.

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Key Features

*   **Real-time Spam Detection:** Monitors Stack Exchange for spam in real-time.
*   **Automated Chatroom Posting:**  Posts detected spam to chatrooms for community awareness.
*   **Integration with Stack Exchange:** Leverages the Stack Exchange API and ChatExchange for data access and communication.
*   **Flexible Setup:** Offers multiple setup options, including virtual environments and Docker containers.

## Getting Started

### Basic Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Checkout the deployment branch: `git checkout deploy`
4.  Install dependencies:
    ```bash
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
5.  Configure the application: Copy `config.sample` to `config` and edit the required values.
6.  Run SmokeDetector: `python3 nocrash.py` (preferably in a daemon-able mode like `screen`).

### Virtual Environment Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Configure git:
    ```bash
    git config user.email "smokey@erwaysoftware.com"
    git config user.name "SmokeDetector"
    ```
4.  Checkout the deployment branch: `git checkout deploy`
5.  Create and activate a virtual environment:
    ```bash
    python3 -m venv env
    ```
6.  Install dependencies:
    ```bash
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
7.  Configure the application: Copy `config.sample` to `config` and edit the required values.
8.  Run SmokeDetector: `env/bin/python3 nocrash.py`

### Docker Setup

1.  Build the Docker image:
    ```bash
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  Create a container:
    ```bash
    docker create --name=mysmokedetector smokey:$DATE
    ```
3.  Copy the configuration file into the container:  Create the `config` file and then:
    ```bash
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```
4.  Optional: Enter the container for further configuration
    ```bash
    docker exec -it mysmokedetector bash
    ```
5.  Indicate when setup is complete
    ```bash
    touch ~smokey/ready
    ```

#### Automate Docker deployment with Docker Compose

Create a directory, place the `config` file and [`docker-compose.yml` file](docker-compose.yml).
Run `docker-compose up -d`

To improve performance, edit `docker-compose.yml` and add the following keys to `smokey`.

```yaml
restart: always  # when your host reboots Smokey can autostart
mem_limit: 512M
cpus: 0.5  # Recommend 2.0 or more for spam waves
```

## Documentation

*   User documentation is available in the [wiki](https://charcoal-se.org/smokey).
*   Detailed setup instructions are in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

## Requirements

*   Stack Exchange login is required.
*   Supports Python versions currently in the [supported phase](https://devguide.python.org/versions/).
*   Git 1.8 or higher (2.11+ recommended) for blacklist/watchlist modifications.

## Blacklist Removal

For requests to remove websites from the blacklist, please refer to "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)".

## License

This project is licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) and MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>) at your option.