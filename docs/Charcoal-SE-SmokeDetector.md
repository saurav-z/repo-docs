# SmokeDetector: Real-Time Spam Detection for Stack Exchange

SmokeDetector is a powerful, headless chatbot designed to identify and report spam on the Stack Exchange network in real-time, providing valuable insights and protection for the community. Check out the original repository for more details: [Charcoal-SE/SmokeDetector](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

## Key Features:

*   **Real-Time Spam Detection:** Monitors Stack Exchange activity for spam and malicious content.
*   **Automated Reporting:** Posts detected spam to chatrooms, alerting moderators and users.
*   **Uses ChatExchange and SE API:** Leverages these tools to analyze content and interact with the Stack Exchange platform.
*   **Flexible Deployment:** Supports various setup methods, including standard, virtual environment, and Docker.
*   **Customizable Configuration:** Easily configure the bot to fit your specific needs.

## Documentation

Comprehensive user documentation is available in the [wiki](https://charcoal-se.org/smokey).

## Installation & Setup

Detailed instructions for setting up and running SmokeDetector can be found in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

### Basic Setup

1.  **Clone the Repository:** `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  **Navigate to Directory:** `cd SmokeDetector`
3.  **Checkout Deploy:** `git checkout deploy`
4.  **Install Dependencies:**
    ```shell
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
5.  **Configure:** Copy `config.sample` to `config` and edit the values.
6.  **Run:** `python3 nocrash.py` (recommended) or `python3 ws.py`.

### Virtual Environment Setup

1.  **Clone the Repository:** `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  **Navigate to Directory:** `cd SmokeDetector`
3.  **Set Git Config:**
    ```shell
    git config user.email "smokey@erwaysoftware.com"
    git config user.name "SmokeDetector"
    ```
4.  **Checkout Deploy:** `git checkout deploy`
5.  **Create and Activate Environment:**
    ```shell
    python3 -m venv env
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
6.  **Configure:** Copy the config file and edit as said above.
7.  **Run:** `env/bin/python3 nocrash.py`.

### Docker Setup

1.  **Build the Image:**
    ```shell
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  **Create a Container:** `docker create --name=mysmokedetector smokey:$DATE`
3.  **Configure:** Copy `config.sample` to `config`, edit the values and then copy to container: `docker cp config mysmokedetector:/home/smokey/SmokeDetector/config`
4.  **Optional: SSH, Git etc:** `docker exec -it mysmokedetector bash` and then `touch ~smokey/ready`
5.  **Docker Compose:** Configure with a docker-compose.yml and a config file. Run `docker-compose up -d`.

## Requirements

*   Stack Exchange Account
*   Supported Python versions (as defined by Python lifecycle: first release to end of life)
*   Git 1.8 or higher (2.11+ recommended)

## Blacklist Removal

If you are an official representative and want to remove a website/product, see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for details.

## License

Licensed under either of:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE)
    or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT)
    or <https://opensource.org/licenses/MIT>)

### Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.