# SmokeDetector: Real-time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot that actively identifies and reports spam on the Stack Exchange network, protecting the community from malicious content.** For more information, see the original repository: [https://github.com/Charcoal-SE/SmokeDetector](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector uses advanced techniques to monitor Stack Exchange in real-time and alert moderators and the community to spam activity.

**Key Features:**

*   **Real-time Spam Detection:**  Monitors the Stack Exchange platform for spam in real-time.
*   **Automated Reporting:** Posts identified spam to designated chatrooms for immediate review.
*   **ChatExchange Integration:** Leverages the ChatExchange library for seamless integration with Stack Exchange chat.
*   **API Driven:** Utilizes the Stack Exchange API to access and analyze content.
*   **Configurable:** Designed to be easily configured to meet the needs of different Stack Exchange communities.
*   **Multiple Deployment Options:** Supports setup via git clone, virtual environments, and Docker for flexible deployment.

**How it Works:**

SmokeDetector operates as a headless chatbot. It accesses the Stack Exchange [realtime tab](https://stackexchange.com/questions?tab=realtime) to collect questions, it uses the [Stack Exchange API](https://api.stackexchange.com/) for answer data, and uses [ChatExchange](https://github.com/Manishearth/ChatExchange) for posting to chatrooms.

**Example Chat Post:**

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Setup and Usage

Detailed documentation can be found in the [wiki](https://charcoal-se.org/smokey).

### Basic Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Switch to the deploy branch: `git checkout deploy`
4.  Install dependencies: `sudo pip3 install -r requirements.txt --upgrade` and `pip3 install --user -r user_requirements.txt --upgrade`
5.  Copy the configuration sample: Copy `config.sample` to a new file called `config`, and edit the values required.
6.  Run SmokeDetector: `python3 nocrash.py` (recommended) or `python3 ws.py`

### Virtual Environment Setup

1.  Follow steps 1-4 above.
2.  Create a virtual environment: `python3 -m venv env`
3.  Activate the virtual environment: (command depends on your shell)
4.  Install dependencies within the environment: `env/bin/pip3 install -r requirements.txt --upgrade` and `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
5.  Copy and configure the `config` file as above.
6.  Run SmokeDetector: `env/bin/python3 nocrash.py`

### Docker Setup

1.  Build the Docker image:
    ```shell
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  Create a container: `docker create --name=mysmokedetector smokey:$DATE`
3.  Copy configuration file into the container: `docker cp config mysmokedetector:/home/smokey/SmokeDetector/config` (after editing the config file)
4.  Optionally, enter the container for further setup: `docker exec -it mysmokedetector bash`, then touch `/home/smokey/ready` after you're ready.
5.  Automate deployment: use `docker-compose` and the provided [`docker-compose.yml` file](docker-compose.yml).  Recommend adding `restart: always`, `mem_limit: 512M`, and `cpus: 0.5` to `docker-compose.yml` for best results.

## Requirements

*   Stack Exchange Login
*   Supported Python versions (see [Python life cycle](https://devguide.python.org/versions/))
*   Git 1.8 or higher (2.11+ recommended)

## Blacklist Removal

Please see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for details.

## License

Licensed under either the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) or the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>) at your option.

### Contribution Licensing

Contributions are dual-licensed under the same terms.