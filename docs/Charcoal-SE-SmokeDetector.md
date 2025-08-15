# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Keep your Stack Exchange communities clean with SmokeDetector, a powerful, open-source chatbot that identifies and flags spam in real-time.**  [See the original repo](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector is a headless chatbot designed to monitor Stack Exchange and immediately flag spam content. Utilizing the Stack Exchange API and ChatExchange, SmokeDetector analyzes real-time posts from the Stack Exchange questions feed and posts spam detections to designated chatrooms.

## Key Features

*   **Real-Time Spam Detection:** Monitors Stack Exchange for spam.
*   **Chat Integration:** Posts spam detections directly to chatrooms.
*   **Open Source:** Fully transparent and community-driven.
*   **Flexible Setup:** Supports various setup methods, including:
    *   **Basic Setup:** Simple installation with `git clone` and `pip3 install`.
    *   **Virtual Environment Setup:** Isolates dependencies for cleaner operation.
    *   **Docker Setup:** Containerizes SmokeDetector for easy deployment.
    *   **Docker Compose** Automates Docker deployment
*   **Customizable:**  Easily configurable via `config` file.

## Getting Started

### Basic Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Checkout deploy: `git checkout deploy`
4.  Install dependencies: `sudo pip3 install -r requirements.txt --upgrade` and `pip3 install --user -r user_requirements.txt --upgrade`
5.  Copy and configure the sample config: Copy `config.sample` to a new file called `config`, and edit the values required.
6.  Run SmokeDetector: `python3 nocrash.py` (recommended) or `python3 ws.py`

### Virtual Environment Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Git configuration: `git config user.email "smokey@erwaysoftware.com"` and `git config user.name "SmokeDetector"`
4.  Checkout deploy: `git checkout deploy`
5.  Create virtual environment: `python3 -m venv env`
6.  Install dependencies: `env/bin/pip3 install -r requirements.txt --upgrade` and `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
7.  Copy and configure the sample config: Copy `config.sample` to a new file called `config`, and edit the values required.
8.  Run SmokeDetector: `env/bin/python3 nocrash.py`

### Docker Setup

1.  **Build the Docker image:**
    ```shell
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```

2.  **Create a Docker container:**
    ```shell
    docker create --name=mysmokedetector smokey:$DATE
    ```

3.  **Configure SmokeDetector:**  Copy `config.sample` to a new file named `config`, edit the values, and copy into container:
    ```shell
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```

4.  **Start the container:** Start the container
5.  **Optional Bash access:** `docker exec -it mysmokedetector bash`
6.  **Ready:** `touch ~smokey/ready`

### Automate Docker deployment with Docker Compose

1.  Ensure that you have a properly filled `config` file. You can start with [the sample](config.sample).
2.  Create a directory.
3.  Place the `config` file and [`docker-compose.yml` file](docker-compose.yml).
4.  Run `docker-compose up -d` and your SmokeDetector instance is up.
5.  If you want additional control like memory and CPU constraint, you can edit `docker-compose.yml` and add the following keys to `smokey`. The example values are recommended values.

```yaml
restart: always  # when your host reboots Smokey can autostart
mem_limit: 512M
cpus: 0.5  # Recommend 2.0 or more for spam waves
```

## Requirements

*   Stack Exchange account.
*   Python versions that are in the [supported phase of the Python life cycle](https://devguide.python.org/versions/).
*   Git 1.8+ (Git 2.11+ recommended) for blacklist/watchlist modifications.

## Resources

*   **User Documentation:** [Wiki](https://charcoal-se.org/smokey)
*   **Setup Instructions:** [Set-Up-and-Run-SmokeDetector](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector)
*   **Process for Blacklist Removal:** [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)

## License

SmokeDetector is licensed under either the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) or the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>) at your option.