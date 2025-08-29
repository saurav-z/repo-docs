# SmokeDetector: Real-Time Spam Detection for Stack Exchange

Tired of spam on Stack Exchange? **SmokeDetector is a powerful, headless chatbot that automatically identifies and reports spam in real-time, keeping your communities clean.**

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector is a Python-based application designed to combat spam on Stack Exchange platforms. It monitors the real-time feed, analyzes content, and posts identified spam to designated chatrooms, alerting moderators and users. This project utilizes the [ChatExchange](https://github.com/Manishearth/ChatExchange) library, accesses question data from the Stack Exchange [realtime tab](https://stackexchange.com/questions?tab=realtime), and retrieves answers via the [Stack Exchange API](https://api.stackexchange.com/).

**Key Features:**

*   **Real-time Spam Detection:** Monitors Stack Exchange for spam in real-time.
*   **Automated Reporting:** Posts detected spam to chatrooms for immediate review.
*   **API Integration:** Leverages the Stack Exchange API for comprehensive data analysis.
*   **Customizable:** Easy to set up and configure to fit your community's needs.
*   **Docker Support**: Supports running the application in a Docker container.

See an example of a [chat post](https://chat.stackexchange.com/transcript/message/43579469) made by SmokeDetector.

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Getting Started

Comprehensive documentation and setup guides are available in the [wiki](https://charcoal-se.org/smokey). Detailed documentation for [setting up and running SmokeDetector is in the wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

### Installation

Choose your preferred setup method below, depending on if you would like to run in a basic environment, virtual environment, or a Docker container.

#### Basic Setup

```shell
git clone https://github.com/Charcoal-SE/SmokeDetector.git
cd SmokeDetector
git checkout deploy
sudo pip3 install -r requirements.txt --upgrade
pip3 install --user -r user_requirements.txt --upgrade
```

*   Copy `config.sample` to a new file called `config`.
*   Edit the values in `config` as needed.
*   Run using `python3 nocrash.py` or `python3 ws.py`.

#### Virtual Environment Setup

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

*   Copy `config.sample` to a new file called `config`.
*   Edit the values in `config` as needed.
*   Run with `env/bin/python3 nocrash.py`.

#### Docker Setup

1.  Build a Docker image:

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

3.  Edit the configuration file and copy it into the container:

```shell
docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
```

4.  Start the container and run SmokeDetector.

You can also use Docker Compose.  Use the `docker-compose.yml` file provided.

```yaml
restart: always  # when your host reboots Smokey can autostart
mem_limit: 512M
cpus: 0.5  # Recommend 2.0 or more for spam waves
```

## Requirements

*   Stack Exchange login credentials.
*   Supported Python versions (refer to documentation for compatibility details).
*   Git 1.8 or higher (Git 2.11+ recommended) for blacklist/watchlist modifications.

## Blacklist Removal

If you're an official representative of a website and want to request removal from the blacklist, please see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for instructions.

## License

This project is available under the terms of the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) or the [MIT license](https://opensource.org/licenses/MIT).

### Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.

For more information and the source code, visit the [SmokeDetector repository on GitHub](https://github.com/Charcoal-SE/SmokeDetector).