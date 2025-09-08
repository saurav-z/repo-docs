# SmokeDetector: Real-time Spam Detection for Stack Exchange

Tired of spam polluting your Stack Exchange communities? **SmokeDetector is a headless chatbot designed to automatically detect and report spam in real-time.** ([See the original repo](https://github.com/Charcoal-SE/SmokeDetector))

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector utilizes the Stack Exchange API and monitors the realtime tab for new questions, posting detected spam to designated chatrooms.  See an example [chat post](https://chat.stackexchange.com/transcript/message/43579469):

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Key Features

*   **Real-time Spam Detection:** Automatically identifies and reports spam.
*   **Headless Chatbot:** Operates seamlessly in the background.
*   **Stack Exchange Integration:** Leverages the Stack Exchange API and realtime data.
*   **Chatroom Reporting:** Posts spam reports to designated chatrooms.
*   **Multiple Deployment Options:**  Offers flexible setup through various methods.

## Getting Started

Comprehensive documentation is available on the [SmokeDetector Wiki](https://charcoal-se.org/smokey).  Detailed setup instructions can be found on the [Setup and Run page](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

### Installation Options

*   **Basic Setup:** Clone the repository, install requirements, configure the `config` file, and run using `python3 nocrash.py`.
*   **Virtual Environment Setup:**  Create an isolated environment to manage dependencies.
*   **Docker Setup:** Use Docker containers for a streamlined and isolated deployment.

###  Docker Compose

Automate your Docker deployment by creating a directory, adding the `config` file, and [`docker-compose.yml` file](docker-compose.yml), and run `docker-compose up -d`.

## Requirements

*   Python versions within the [supported phase of the Python life cycle](https://devguide.python.org/versions/)
*   Stack Exchange Login
*   Git 1.8 or higher (2.11+ recommended) for blacklist/watchlist modifications.

## Blacklist Removal

For website/product removal requests from the blacklist, see the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for details.

## License

SmokeDetector is available under the following licenses:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

### Contribution Licensing

Contributions are dual-licensed under the Apache-2.0 license.