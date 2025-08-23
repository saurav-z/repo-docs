# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Instantly identify and report spam on Stack Exchange with SmokeDetector, a powerful headless chatbot.**  View the original repository [here](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector is a headless chatbot designed to detect and report spam within the Stack Exchange network. It analyzes questions from the [Stack Exchange realtime tab](https://stackexchange.com/questions?tab=realtime) and utilizes the [Stack Exchange API](https://api.stackexchange.com/) for data. SmokeDetector posts detected spam to chatrooms, enabling rapid identification and moderation.

Example [chat post](https://chat.stackexchange.com/transcript/message/43579469):

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Key Features

*   **Real-time Spam Detection:** Monitors the Stack Exchange network for spam content.
*   **Automated Reporting:** Posts spam reports to designated chatrooms for immediate action.
*   **API Integration:** Leverages the Stack Exchange API for data access.
*   **Customizable Configuration:** Allows configuration to fit your needs.
*   **Multiple Deployment Options:** Supports setup via standard installation, virtual environments, and Docker containers.

## Getting Started

### Basic Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate into the directory: `cd SmokeDetector`
3.  Switch to the deploy branch: `git checkout deploy`
4.  Install required packages (using `pip3`, and your chosen virtual environment if you're using one):
    ```bash
    sudo pip3 install -r requirements.txt --upgrade  # or `env/bin/pip3` if using a virtual environment
    pip3 install --user -r user_requirements.txt --upgrade # or `env/bin/pip3` if using a virtual environment
    ```
5.  Copy the sample configuration file: `cp config.sample config`
6.  Edit the `config` file with your specific credentials and settings.
7.  Run SmokeDetector: `python3 nocrash.py` (recommended for continuous operation) or `python3 ws.py` (shuts down after 6 hours).

### Virtual Environment Setup

For a cleaner setup:

1.  Follow steps 1-3 from Basic Setup.
2.  Create a virtual environment: `python3 -m venv env`
3.  Activate the environment: (varies by system, consult your terminal's documentation)
4.  Install dependencies: `env/bin/pip3 install -r requirements.txt --upgrade` and `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
5.  Copy and configure the `config` file (as in Basic Setup).
6.  Run SmokeDetector: `env/bin/python3 nocrash.py`.

### Docker Setup

For optimal isolation and ease of deployment:

1.  Grab the [Dockerfile](Dockerfile) and build an image:
    ```bash
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  Create a container: `docker create --name=mysmokedetector smokey:$DATE`
3.  Copy and configure the `config` file:
    ```bash
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```
4.  (Optional) Enter the container for further setup (e.g., SSH, Git): `docker exec -it mysmokedetector bash` and create a `ready` file.
5.  (Automated setup is also available using Docker Compose - see the original documentation).

## Requirements

*   **Stack Exchange Account:** You'll need valid Stack Exchange credentials.
*   **Python:** Supports Python versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/).  We run CI testing on that span of versions.  It is known to not work on Python 3.6 and lower.
*   **Git:** Git 1.8+ (2.11+ recommended) is required for blacklist/watchlist modifications.

## Blacklist Removal

If you are a website representative and would like to request removal from the blacklist, see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for detailed instructions.

## License

SmokeDetector is licensed under the following, at your option:

*   Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

### Contribution Licensing

By submitting your contribution for inclusion in the work, you agree that it be dual licensed as above, without any additional terms or conditions.