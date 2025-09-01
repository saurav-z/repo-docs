# SmokeDetector: Real-time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, open-source, headless chatbot that instantly identifies and reports spam across the Stack Exchange network.** Find the original repo at: [https://github.com/Charcoal-SE/SmokeDetector](https://github.com/Charcoal-SE/SmokeDetector).

<br>

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

<br>

## Key Features

*   **Real-time Spam Detection:** Monitors the Stack Exchange network in real-time to quickly identify and report spam.
*   **Automated Reporting:** Posts detected spam to chatrooms for community review and action.
*   **Integration with Stack Exchange:** Leverages the Stack Exchange API and utilizes the real-time question feed.
*   **Configurable:** Easily set up and customize SmokeDetector to fit your needs.
*   **Open Source:** Freely available under the Apache 2.0 and MIT licenses, encouraging community contributions.

## How it Works

SmokeDetector utilizes [ChatExchange](https://github.com/Manishearth/ChatExchange) to connect to Stack Exchange chatrooms, accessing question data from the Stack Exchange [realtime tab](https://stackexchange.com/questions?tab=realtime) and the [Stack Exchange API](https://api.stackexchange.com/) to identify spam posts.

## Getting Started

Detailed setup and running instructions can be found in the [wiki](https://charcoal-se.org/smokey).

### Basic Setup

1.  **Clone the Repository:** `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  **Navigate to the Directory:** `cd SmokeDetector`
3.  **Checkout Deploy Branch:** `git checkout deploy`
4.  **Install Dependencies:**

    ```shell
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```

5.  **Configure:** Copy `config.sample` to `config` and edit the necessary values.
6.  **Run:** `python3 nocrash.py` (recommended for continuous operation).

### Virtual Environment Setup

Using a virtual environment is highly recommended.

1.  **Create Virtual Environment:**

    ```shell
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git config user.email "smokey@erwaysoftware.com"
    git config user.name "SmokeDetector"
    git checkout deploy

    python3 -m venv env
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```

2.  **Configure:** Copy `config.sample` to `config` and edit the necessary values.
3.  **Run:** `env/bin/python3 nocrash.py`

### Docker Setup

Docker provides the most isolated and robust deployment.

1.  **Build the Docker Image:**

    ```shell
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```

2.  **Create a Container:** `docker create --name=mysmokedetector smokey:$DATE`
3.  **Configure:** Copy `config.sample` to `config`, edit it, and then copy it into the container: `docker cp config mysmokedetector:/home/smokey/SmokeDetector/config`
4.  **Run:** Start the container.

## Advanced Setup & Deployment

*   **Docker Compose:**  See the `docker-compose.yml` file for automating Docker deployments, and recommendations for resource allocation.
*   **Additional Docker Configuration:**  Use `docker exec -it mysmokedetector bash` for further customization.
*   **Configuration File:**  Ensure the `config` file is properly configured.

## Requirements

*   Stack Exchange Login
*   Supported Python version (as defined by the Python life cycle).
*   Git 1.8+ (recommended: 2.11+) for blacklist/watchlist contributions.

## Blacklist Removal Requests

For removal requests, please refer to the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" documentation.

## License

SmokeDetector is licensed under the following terms:

*   Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)