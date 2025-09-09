# SmokeDetector: Real-time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot that tirelessly identifies and reports spam on Stack Exchange, ensuring a cleaner and more user-friendly experience.**

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector uses [ChatExchange](https://github.com/Manishearth/ChatExchange) to monitor the Stack Exchange [realtime tab](https://stackexchange.com/questions?tab=realtime) and utilizes the [Stack Exchange API](https://api.stackexchange.com/) to analyze content for spam.

**Key Features:**

*   **Real-time Spam Detection:** Identifies spam posts quickly and efficiently.
*   **Automated Reporting:** Posts spam reports to designated chatrooms.
*   **Flexible Setup:** Supports various setup methods including basic, virtual environment, and Docker.
*   **Customizable:** Configuration options allow for tailored spam detection rules.
*   **Open Source:** Built by the community, see the [original repo](https://github.com/Charcoal-SE/SmokeDetector) for more information.

**Example Chat Post:**

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Documentation

*   Comprehensive user documentation is available in the [wiki](https://charcoal-se.org/smokey).
*   Detailed setup and running instructions are in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

## Installation

Choose your preferred installation method:

### Basic Setup

```shell
git clone https://github.com/Charcoal-SE/SmokeDetector.git
cd SmokeDetector
git checkout deploy
sudo pip3 install -r requirements.txt --upgrade
pip3 install --user -r user_requirements.txt --upgrade
```

*   Copy `config.sample` to `config` and edit the necessary values.
*   Run with `python3 nocrash.py` (recommended for daemon mode) or `python3 ws.py` (will shut down after 6 hours).

### Virtual Environment Setup

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

*   Copy and edit the config file.
*   Run with `env/bin/python3 nocrash.py`.

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

3.  Copy your `config` file into the container.

    ```shell
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```

4.  Optionally, access the container with a Bash shell.

    ```shell
    docker exec -it mysmokedetector bash
    ```

5.  Create `ready` file in home directory.

    ```shell
    touch ~smokey/ready
    ```

#### Automate Docker deployment with Docker Compose

Create a directory and place `config` and [`docker-compose.yml`](docker-compose.yml) in it.
Run `docker-compose up -d`.

Adjust memory and CPU constraints within `docker-compose.yml` if required.

```yaml
restart: always
mem_limit: 512M
cpus: 0.5
```

## Requirements

*   Stack Exchange login required.
*   Supports Python versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/).
*   Git 1.8+ is recommended (2.11+).

## Blacklist Removal

If you are an official representative of a website wishing to be removed from the blacklist, please see the [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal).

## License

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT license](LICENSE-MIT).

### Contribution Licensing

Contributions are dual-licensed under the same terms as the project.