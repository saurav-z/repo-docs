# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Stop spam in its tracks! SmokeDetector is a powerful, headless chatbot that automatically identifies and reports malicious content on Stack Exchange.**  [Explore the original repository](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector utilizes the Stack Exchange API and real-time data streams to detect spam and malicious content, reporting it directly to chatrooms.

**Key Features:**

*   **Automated Spam Detection:** Identifies spam and harmful content using a variety of signals and rules.
*   **Real-Time Reporting:**  Posts detected spam to designated chatrooms for immediate action.
*   **Open Source:** Freely available and customizable, allowing for community contributions and improvements.
*   **Robust Technology:** Built using [ChatExchange](https://github.com/Manishearth/ChatExchange) and designed to be reliable and scalable.
*   **Flexible Deployment:** Supports various setup options, including standard, virtual environment, and Docker.

Example [chat post](https://chat.stackexchange.com/transcript/message/43579469):

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Getting Started

Detailed instructions and setup guides are available in the [wiki](https://charcoal-se.org/smokey).

### Installation Methods

Choose your preferred method:

*   **Basic Setup:** Clone the repository, install dependencies using `pip`, configure the `config` file, and run `python3 nocrash.py`.
*   **Virtual Environment Setup:** Recommended for isolating dependencies. Follow similar steps to the basic setup, but activate your virtual environment before installing dependencies and running SmokeDetector.
*   **Docker Setup:** Provides the most isolated and reproducible environment. Follow the Dockerfile instructions to build and run the container.  Automate Docker deployment with Docker Compose

## Requirements

*   Stack Exchange Login
*   Supported Python versions (See the original README).
*   Git 1.8 or higher (Git 2.11+ recommended) is required to commit Blacklist/Watchlist changes to GitHub.

## Blacklist Removal

For official representatives seeking blacklist removal, please see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for detailed steps.

## License

This project is licensed under either the:

*   Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.