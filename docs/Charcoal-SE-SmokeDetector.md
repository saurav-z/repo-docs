# SmokeDetector: Real-time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot that tirelessly monitors Stack Exchange for spam, immediately posting detections to designated chatrooms.** ([View the original repository](https://github.com/Charcoal-SE/SmokeDetector))

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector uses the Stack Exchange API and ChatExchange to analyze questions from the Stack Exchange realtime tab and identify spam, providing instant alerts.

**Key Features:**

*   **Real-time Spam Detection:** Identifies and reports spam as it appears on Stack Exchange.
*   **Automated Chat Posting:** Automatically posts spam detections to connected chatrooms.
*   **Flexible Setup:** Offers multiple setup options, including basic setup, virtual environment, and Docker.
*   **Configurable:** Allows customization via a configuration file.
*   **Supports Latest Python Versions:**  Continuously tested to support all Python versions currently in the supported phase.
*   **Blacklist Management:**  Includes a process for requesting removal of websites from the blacklist.

**Example Chat Post:**

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Installation and Setup

Detailed documentation for setting up and running SmokeDetector is available in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector). The following are installation steps:

### Basic Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Checkout the deploy branch: `git checkout deploy`
4.  Install dependencies: `sudo pip3 install -r requirements.txt --upgrade` and `pip3 install --user -r user_requirements.txt --upgrade`
5.  Copy the configuration file: Copy `config.sample` to a new file called `config`
6.  Edit the configuration: Modify the necessary values in the `config` file.
7.  Run SmokeDetector: Execute `python3 nocrash.py` (recommended for persistent operation).

### Virtual Environment Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Configure git: Configure your user name and email for Git
4.  Checkout the deploy branch: `git checkout deploy`
5.  Create the virtual environment: `python3 -m venv env`
6.  Install dependencies in the virtual environment: `env/bin/pip3 install -r requirements.txt --upgrade` and `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
7.  Copy and edit the configuration file (as above).
8.  Run SmokeDetector: Execute `env/bin/python3 nocrash.py`.

### Docker Setup

1.  Follow the instructions in the [Dockerfile](Dockerfile) for building a Docker image.
2.  Create a container from the image.
3.  Copy and configure the `config` file within the container.
4.  Start the container and run `SmokeDetector`.

#### Automate Docker deployment with Docker Compose

1.  Create a directory and place your filled `config` file and [`docker-compose.yml` file](docker-compose.yml).
2.  Run `docker-compose up -d` to start your SmokeDetector instance.
3.  Customize resources in the `docker-compose.yml` file (e.g., `mem_limit`, `cpus`).

## Requirements

*   Stack Exchange login.
*   Supported Python version (see above).
*   Git 1.8 or higher (2.11+ recommended) for modifying the blacklist and watchlist.

## Requesting Blacklist Removal

Please see the [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal) if you are the official representative of a website/product and wish to request its removal.

## License

SmokeDetector is licensed under the Apache License, Version 2.0, or the MIT license, at your option. See [LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0> and [LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT> for more information.