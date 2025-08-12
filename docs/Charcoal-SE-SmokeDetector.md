# SmokeDetector: Real-Time Spam Detection for Stack Exchange

**Tired of spam cluttering your Stack Exchange communities? SmokeDetector is a powerful, headless chatbot designed to identify and report spam in real-time.**

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector is a headless chatbot that monitors Stack Exchange's real-time feed and uses a rules engine to detect spam and other unwanted content, posting alerts to chatrooms.  It leverages the [ChatExchange](https://github.com/Manishearth/ChatExchange) library, the Stack Exchange [realtime tab](https://stackexchange.com/questions?tab=realtime), and the Stack Exchange [API](https://api.stackexchange.com/) to achieve this.

[See SmokeDetector in action!](https://chat.stackexchange.com/transcript/message/43579469)

![Example chat post](https://i.sstatic.net/oLyfb.png)

**Key Features:**

*   **Real-time Spam Detection:** Identifies spam and undesirable content as it appears on Stack Exchange.
*   **Automated Reporting:** Posts alerts to designated chatrooms, enabling moderators and community members to take swift action.
*   **Customizable:** Configurable rules engine allows for tailored spam detection based on community needs.
*   **Flexible Deployment:** Supports various deployment methods, including direct execution, virtual environments, and Docker containers.

## Installation and Setup

Detailed documentation for setting up and running SmokeDetector is available in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

### Quick Start (General)

1.  **Clone the Repository:** `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  **Navigate:** `cd SmokeDetector`
3.  **Checkout Deploy Branch:** `git checkout deploy`
4.  **Install Dependencies:** `sudo pip3 install -r requirements.txt --upgrade && pip3 install --user -r user_requirements.txt --upgrade`
5.  **Configure:** Copy `config.sample` to `config` and edit the values to match your environment.
6.  **Run:** `python3 nocrash.py` (recommended) or `python3 ws.py`

### Advanced Setup Options

*   **Virtual Environment:** Isolate dependencies using Python's `venv` module.
*   **Docker:** Containerize SmokeDetector for improved portability and resource management. Instructions are detailed in the README.
*   **Docker Compose:** Simplifies Docker deployments with an example `docker-compose.yml` configuration.

## Requirements

*   Stack Exchange Login
*   Supported Python versions: (See Python life cycle.)
*   Git 1.8+ (Git 2.11+ recommended)

## Blacklist Removal

For official representatives of websites seeking removal from the blacklist, please refer to the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for details.

## License

SmokeDetector is available under the terms of the:

*   Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

See the `LICENSE-APACHE` and `LICENSE-MIT` files for more details.

## Contributing

See the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0) for contribution licensing details.

---
**[Visit the SmokeDetector Repository on GitHub](https://github.com/Charcoal-SE/SmokeDetector)**