# SmokeDetector: The Ultimate Spam Detection Bot for Stack Exchange

**Tired of spam cluttering your Stack Exchange chatrooms? SmokeDetector is a powerful, open-source chatbot that automatically identifies and flags spam, keeping your community clean.**  ([Original Repository](https://github.com/Charcoal-SE/SmokeDetector))

## Key Features:

*   **Real-time Spam Detection:**  Monitors the Stack Exchange realtime tab for new questions.
*   **Automated Reporting:** Posts suspected spam to chatrooms for review.
*   **Community-Driven:** Relies on a blacklist and watchlist maintained by the community to identify malicious content.
*   **Configurable:** Easily customizable to fit your specific needs and community rules.
*   **Flexible Deployment:** Supports various setup methods, including:
    *   **Standard Installation:**  Uses `git`, `pip3`, and a `config` file.
    *   **Virtual Environment:**  Isolates dependencies using `venv`.
    *   **Docker Container:**  Provides a containerized environment for easy deployment and management.
    *   **Docker Compose:** Automate Docker deployment with docker-compose

## Getting Started

### Installation (Standard)

1.  Clone the repository:

    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git checkout deploy
    ```

2.  Install dependencies:

    ```bash
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```

3.  Configure SmokeDetector:

    *   Copy `config.sample` to `config`.
    *   Edit the `config` file with your specific details.

4.  Run SmokeDetector:

    ```bash
    python3 nocrash.py
    ```
    *(Run in a daemon-able mode, like a `screen` session.)*

### Virtual Environment Setup

1.  Clone the repository:

    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git config user.email "smokey@erwaysoftware.com"
    git config user.name "SmokeDetector"
    git checkout deploy
    ```

2.  Create and activate a virtual environment:

    ```bash
    python3 -m venv env
    ```

3.  Install dependencies:

    ```bash
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```

4.  Configure SmokeDetector (same as standard installation).
5.  Run SmokeDetector:
    ```bash
    env/bin/python3 nocrash.py
    ```

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

3.  Configure SmokeDetector:
    *   Copy `config.sample` to `config`.
    *   Edit the `config` file with your specific details.
    *   Copy the `config` file into the container:

    ```bash
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```

4.  Start the container and make sure the bot is ready with the `ready` file under `/home/smokey`.

### Docker Compose Setup

1.  Create a `config` file based on `config.sample`.
2.  Place `config` file and the `docker-compose.yml` file in a directory.
3.  Run `docker-compose up -d`.

#### Docker Compose Optimization

Customize `docker-compose.yml` to set:

*   `restart: always` to autostart on host reboot.
*   `mem_limit` and `cpus` to manage resource usage, if required.

## Requirements

*   Stack Exchange Login
*   [Supported Python versions](https://devguide.python.org/versions/) (defined as between "First release" and "End of life").
*   Git 1.8+ (2.11+ recommended) for blacklist/watchlist modifications.

## Blacklist Removal

For website/product blacklist removal requests, please refer to:
[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)

## License

SmokeDetector is licensed under the Apache License, Version 2.0 or the MIT license. See the LICENSE-APACHE and LICENSE-MIT files or the license links for details.