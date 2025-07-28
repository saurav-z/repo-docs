# SmokeDetector: Real-time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot that identifies and reports spam on the Stack Exchange network in real-time.**  Find the original repository [here](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector utilizes the Stack Exchange API and ChatExchange to monitor the Stack Exchange [realtime tab](https://stackexchange.com/questions?tab=realtime) and flag potential spam, posting alerts to designated chatrooms.

**Key Features:**

*   **Real-time Spam Detection:**  Quickly identifies and reports spam posts.
*   **Automated Chatroom Alerts:** Posts spam reports to specified chatrooms for moderation.
*   **Uses Stack Exchange API:** Leverages the official API for data access.
*   **Flexible Deployment:** Supports various setup methods, including virtual environments and Docker.
*   **Extensive Documentation:** Comprehensive documentation is available in the [wiki](https://charcoal-se.org/smokey).

## Setup and Installation

Detailed instructions for setting up and running SmokeDetector are available in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).  Here's a summary of the main setup methods:

### Basic Setup

1.  Clone the repository:  `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory:  `cd SmokeDetector`
3.  Checkout the deploy branch: `git checkout deploy`
4.  Install dependencies:
    ```bash
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
5.  Configure SmokeDetector by copying `config.sample` to `config` and editing the required values.
6.  Run SmokeDetector:  `python3 nocrash.py`

### Virtual Environment Setup

1.  Clone the repository:  `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory:  `cd SmokeDetector`
3.  Configure git
    ```bash
    git config user.email "smokey@erwaysoftware.com"
    git config user.name "SmokeDetector"
    ```
4.  Checkout the deploy branch: `git checkout deploy`
5.  Create and activate a virtual environment:
    ```bash
    python3 -m venv env
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
6.  Configure SmokeDetector by copying `config.sample` to `config` and editing the required values.
7.  Run SmokeDetector:  `env/bin/python3 nocrash.py`

### Docker Setup

1.  Build the Docker image:
    ```bash
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  Create a container:  `docker create --name=mysmokedetector smokey:$DATE`
3.  Configure SmokeDetector by copying `config.sample` to `config` and editing the required values inside the container:
    ```bash
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```
4.  Start the container (after setting up the config file).
5.  Optionally, enter the container for additional setup: `docker exec -it mysmokedetector bash`
6.  Create a `ready` file to signal readiness: `touch ~smokey/ready`

#### Automate Docker deployment with Docker Compose

1.  Create a directory and place your `config` file and the `docker-compose.yml` file.
2.  Run: `docker-compose up -d`

You can configure resource limits within `docker-compose.yml`.

## Requirements

*   Stack Exchange Account
*   Supported Python version as per [Python life cycle](https://devguide.python.org/versions/)
*   Git 1.8 or higher (2.11+ recommended) for blacklist/watchlist modifications.

## Requesting Blacklist Removal

For website/product removal requests, please see the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" in the wiki.

## License

This project is available under the following licenses:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE)
    or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT)
    or <https://opensource.org/licenses/MIT>)

### Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.