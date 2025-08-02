# SmokeDetector: Real-Time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot designed to automatically identify and report spam on the Stack Exchange network.** [Check out the original repository](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

## Key Features

*   **Automated Spam Detection:** Identifies spam posts in real-time using advanced algorithms.
*   **Chatroom Reporting:**  Posts spam reports to designated chatrooms for moderation.
*   **Stack Exchange Integration:** Leverages the Stack Exchange API and real-time feed for efficient spam detection.
*   **Flexible Deployment:** Supports various setup methods including standard, virtual environment, and Docker.
*   **Customizable:** Easily configurable through the `config` file to suit specific needs.

## How it Works

SmokeDetector utilizes the [ChatExchange](https://github.com/Manishearth/ChatExchange) library to connect to chatrooms, monitors the Stack Exchange [realtime tab](https://stackexchange.com/questions?tab=realtime) for new questions, and fetches data via the [Stack Exchange API](https://api.stackexchange.com/) to analyze content and identify spam.

**Example Chat Post:**

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Getting Started

### Basic Setup

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
3.  Configure: Copy `config.sample` to a new file named `config` and edit the required values.
4.  Run: Execute `python3 nocrash.py` (recommended for continuous operation).

### Advanced Setup Options

*   **Virtual Environment:** Instructions are provided to set up and run SmokeDetector within a virtual environment for dependency isolation.
*   **Docker:** Detailed steps are included to build, configure, and run SmokeDetector in a Docker container, including Docker Compose automation.

### Deployment with Docker Compose

1.  Ensure you have a properly filled `config` file (start with `config.sample`).
2.  Create a directory, place the `config` file and [`docker-compose.yml` file](docker-compose.yml) in it.
3.  Run `docker-compose up -d` to start SmokeDetector.
4.  Customize the `docker-compose.yml` for resource constraints, using `mem_limit` and `cpus` values.

## Requirements

*   **Stack Exchange Login:** SmokeDetector requires Stack Exchange login credentials.
*   **Python:** Supports Python versions in the [supported phase of the Python lifecycle](https://devguide.python.org/versions/).
*   **Git:** Git 1.8 or higher (Git 2.11+ recommended) is required for blacklist and watchlist modifications.

## Documentation

*   **User Documentation:** Available on the [wiki](https://charcoal-se.org/smokey).
*   **Setup and Run Instructions:**  Find detailed instructions on [setting up and running SmokeDetector in the wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).
*   **Blacklist Removal:**  If you are a website representative, follow the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" to request removal from the blacklist.

## License

SmokeDetector is dual-licensed under the Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) and the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>), at your option.

### Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.