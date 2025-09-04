# SmokeDetector: Real-Time Spam Detection for Stack Exchange

**Detect and report spam on Stack Exchange in real-time with SmokeDetector, a powerful and customizable chatbot.**

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector is a headless chatbot designed to identify and report spam posts on Stack Exchange sites. It monitors the real-time feed and uses the Stack Exchange API to quickly identify and flag potentially malicious content.  [View the original repository here](https://github.com/Charcoal-SE/SmokeDetector).

Key Features:

*   **Real-time Spam Detection:** Monitors the Stack Exchange realtime tab for new questions.
*   **Chat Integration:** Posts detected spam to designated chatrooms.
*   **Stack Exchange API Integration:** Accesses answers and question data through the Stack Exchange API.
*   **Customizable:** Configure to match your needs using the configuration file.
*   **Docker Support:** Ready to be deployed with Docker for isolated and controlled environments.

Example Chat Post:

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Getting Started

Comprehensive documentation, including setup and usage instructions, is available in the [wiki](https://charcoal-se.org/smokey).

### Supported Python Versions
SmokeDetector supports Python versions which are in the [supported phase of the Python life cycle](https://devguide.python.org/versions/) (as defined as between "First release" and "End of life"). We run CI testing on that span of versions. SmokeDetector may work on older versions of Python, but we don't support them and may, at any time, write code that prevents use in older, unsupported versions. We know SmokeDetector is broken on Python 3.6 and lower. While we don't support versions that haven't reached "First release", we're not adverse to hearing about changes in new Python versions which will require us to make changes to SmokeDetector's code, so we can make the transition to supporting new versions of Python smoother.

### Installation

Choose your preferred setup method:

*   **Basic Setup:**
    1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
    2.  Navigate to the directory: `cd SmokeDetector`
    3.  Switch to the deploy branch: `git checkout deploy`
    4.  Install dependencies:
        ```shell
        sudo pip3 install -r requirements.txt --upgrade
        pip3 install --user -r user_requirements.txt --upgrade
        ```
    5.  Copy `config.sample` to a file named `config` and configure the necessary values.
    6.  Run SmokeDetector using: `python3 nocrash.py` (recommended) or `python3 ws.py`.
*   **Virtual Environment Setup:**
    1.  Follow steps 1-4 of the basic setup.
    2.  Create a virtual environment: `python3 -m venv env`
    3.  Install dependencies within the environment:
        ```shell
        env/bin/pip3 install -r requirements.txt --upgrade
        env/bin/pip3 install --user -r user_requirements.txt --upgrade
        ```
    4.  Copy `config.sample` to `config` and configure.
    5.  Run using: `env/bin/python3 nocrash.py`.
*   **Docker Setup:**
    1.  Follow the instructions in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector) to set up SmokeDetector in a Docker container.
    2.  Grab the [Dockerfile](Dockerfile) and build an image of SmokeDetector:

        ```shell
        DATE=$(date +%F)
        mkdir temp
        cd temp
        wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
        docker build -t smokey:$DATE .
        ```
    3.  Create a container from the image you just built

        ```shell
        docker create --name=mysmokedetector smokey:$DATE
        ```
    4.  Start the container.
        Don't worry, SmokeDetector won't run until it's ready,
        so you have the chance to edit the configuration file before SmokeDetector runs.

        Copy `config.sample` to a new file named `config`
        and edit the values required,
        then copy the file into the container with this command:

        ```shell
        docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
        ```
    5.  If you would like to set up additional stuff (SSH, Git etc.),
        you can do so with a Bash shell in the container:

        ```shell
        docker exec -it mysmokedetector bash
        ```

        After you're ready, put a file named `ready` under `/home/smokey`:

        ```shell
        touch ~smokey/ready
        ```
    6.  For Automate Docker deployment with Docker Compose, follow the instructions in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

## Requirements

*   Stack Exchange login credentials are required.
*   Git 1.8 or higher (2.11+ recommended) is needed for blacklisting modifications.
*   Python in [supported phase of the Python life cycle](https://devguide.python.org/versions/) is required.

## Blacklist Removal Requests

If you are an official representative of a website/product and wish to request removal from the blacklist, please refer to the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for detailed instructions.

## License

SmokeDetector is licensed under the following:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

You can choose either license.

### Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.