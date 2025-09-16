# SmokeDetector: Real-Time Spam Detection for Stack Exchange

Tired of spam cluttering your Stack Exchange communities? SmokeDetector, a powerful, headless chatbot, automatically detects and reports spam, keeping your chatrooms clean. Explore the [original repository](https://github.com/Charcoal-SE/SmokeDetector) for the source code.

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

**Key Features:**

*   **Real-time Spam Detection:** Monitors Stack Exchange for spam and malicious content.
*   **Automated Reporting:** Posts detected spam to designated chatrooms for moderation.
*   **Uses Stack Exchange API:** Leverages the Stack Exchange API for efficient data retrieval.
*   **Headless Chatbot:** Operates in the background, requiring no user interaction.
*   **Flexible Deployment:** Supports multiple setup options, including direct install, virtual environment and Docker.

**How SmokeDetector Works:**

SmokeDetector uses [ChatExchange](https://github.com/Manishearth/ChatExchange) to connect to Stack Exchange chatrooms and the Stack Exchange [realtime tab](https://stackexchange.com/questions?tab=realtime) and [Stack Exchange API](https://api.stackexchange.com/) to identify and report potential spam.

**Example Chat Post:**

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Getting Started

Detailed documentation, including setup and configuration instructions, is available in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

### Direct Installation

```shell
git clone https://github.com/Charcoal-SE/SmokeDetector.git
cd SmokeDetector
git checkout deploy
sudo pip3 install -r requirements.txt --upgrade
pip3 install --user -r user_requirements.txt --upgrade
```

1.  Copy `config.sample` to a new file named `config`.
2.  Edit the values in `config` with your desired settings.
3.  Run the bot using `python3 nocrash.py` (recommended for persistent operation, e.g., with `screen`) or `python3 ws.py` (will shut down after 6 hours).

### Virtual Environment Setup

Using a virtual environment is recommended to isolate dependencies.

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

1.  Copy `config.sample` to a new file named `config`.
2.  Edit the values in `config` with your desired settings.
3.  Run SmokeDetector using `env/bin/python3 nocrash.py`.

### Docker Setup

Docker provides an even more isolated environment.

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

3.  Start the container and configure:

    *   Copy `config.sample` to a new file named `config`
    *   Edit the values in `config` with your desired settings.
    *   Copy the `config` file into the container:
        ```shell
        docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
        ```

4.  (Optional) Enter the container for additional setup:

    ```shell
    docker exec -it mysmokedetector bash
    ```
    Then create a `ready` file:
    ```shell
    touch ~smokey/ready
    ```

#### Automate Docker deployment with Docker Compose

1.  Configure a `config` file.
2.  Create a directory with the `config` file and the `docker-compose.yml` file.
3.  Run `docker-compose up -d` to start.

    You can add `restart`, `mem_limit` and `cpus` keys to your `docker-compose.yml` file.

    ```yaml
    restart: always  # when your host reboots Smokey can autostart
    mem_limit: 512M
    cpus: 0.5  # Recommend 2.0 or more for spam waves
    ```

## Requirements

*   Stack Exchange Login
*   [Supported Python versions](https://devguide.python.org/versions/)
*   Git 1.8 or higher (2.11+ recommended) for blacklist/watchlist modifications.

## Blacklist Removal

If you are an official representative, see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for instructions.

## License

SmokeDetector is released under the following licenses:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE)
    or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT)
    or <https://opensource.org/licenses/MIT>)

### Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.