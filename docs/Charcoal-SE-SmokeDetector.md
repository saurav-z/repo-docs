# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Tired of spam on Stack Exchange? SmokeDetector is a powerful headless chatbot that automatically identifies and reports spam in real-time, keeping your community clean.**  You can find the original repository here: [https://github.com/Charcoal-SE/SmokeDetector](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector is a headless chatbot designed to detect and report spam on Stack Exchange in real-time. Leveraging the Stack Exchange API and [ChatExchange](https://github.com/Manishearth/ChatExchange), it monitors the [realtime tab](https://stackexchange.com/questions?tab=realtime) for new questions, analyzes them for spam indicators, and posts detected spam to designated chatrooms.

Here's what SmokeDetector offers:

*   **Automated Spam Detection:** Identifies spam posts using a variety of techniques and criteria.
*   **Real-time Monitoring:** Continuously scans new questions as they appear.
*   **Chatroom Reporting:** Posts spam reports to chatrooms for community awareness and action.
*   **API Integration:** Utilizes the Stack Exchange API to access question data.
*   **Flexible Deployment:** Supports setup via Git, virtual environments, and Docker.

Example [chat post](https://chat.stackexchange.com/transcript/message/43579469):

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Getting Started

Detailed setup and usage instructions are available in the [wiki](https://charcoal-se.org/smokey).  Specific setup guides include:

*   [Setting up and running SmokeDetector](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector)

### Basic Setup

```shell
git clone https://github.com/Charcoal-SE/SmokeDetector.git
cd SmokeDetector
git checkout deploy
sudo pip3 install -r requirements.txt --upgrade
pip3 install --user -r user_requirements.txt --upgrade
```

1.  Copy `config.sample` to a file named `config` and configure the necessary values.
2.  Run SmokeDetector using `python3 nocrash.py` (recommended for daemonized operation) or `python3 ws.py` (shuts down after 6 hours).

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

1.  Copy and configure the `config` file as described above.
2.  Run SmokeDetector using `env/bin/python3 nocrash.py`.

### Docker Setup

1.  **Build the Docker Image:**

    ```shell
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  **Create a Container:**

    ```shell
    docker create --name=mysmokedetector smokey:$DATE
    ```
3.  **Configure and Run:**

    1.  Copy `config.sample` to `config` and edit.
    2.  Copy the `config` file into the container: `docker cp config mysmokedetector:/home/smokey/SmokeDetector/config`
    3.  (Optional) Access the container's shell: `docker exec -it mysmokedetector bash`
    4.  Create a `ready` file: `touch ~smokey/ready`

#### Automate Docker Deployment with Docker Compose

1.  Ensure a properly filled `config` file exists.
2.  Create a directory, place the `config` file and `docker-compose.yml`.
3.  Run `docker-compose up -d`.

    *   For resource control, modify `docker-compose.yml` (example):

    ```yaml
    restart: always
    mem_limit: 512M
    cpus: 0.5
    ```

## Requirements

*   Python versions that are in the [supported phase of the Python life cycle](https://devguide.python.org/versions/)
*   Git 1.8 or higher (Git 2.11+ recommended)

## Blacklist Removal

If you are an official representative and wish to request the removal of a website from the blacklist, please consult the [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal).

## License

This project is licensed under either the:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution Licensing

Contributions are dual-licensed under the same terms as the project.