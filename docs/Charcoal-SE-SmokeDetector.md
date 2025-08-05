# SmokeDetector: Real-Time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot designed to automatically detect and report spam on the Stack Exchange network.** [Check out the original repository here](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector leverages the Stack Exchange API and real-time data to identify and flag potentially malicious content, posting alerts to chatrooms for community review.

**Key Features:**

*   **Real-time Spam Detection:** Monitors Stack Exchange for spam in real-time.
*   **Automated Reporting:** Posts detected spam to chatrooms for community review.
*   **Uses Stack Exchange APIs:** Leverages the Stack Exchange API for accessing and processing data.
*   **Flexible Deployment:** Supports various setup options, including standard, virtual environment, and Docker.
*   **Customizable Configuration:** Allows users to configure settings and tailor the bot's behavior.

**Getting Started:**

Detailed setup instructions and documentation can be found in the [wiki](https://charcoal-se.org/smokey) including:
*   [Setting Up and Running SmokeDetector](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector)

**Setup Options:**

*   **Basic Setup:** Instructions to install and run with `pip3` (see original README for more details).
*   **Virtual Environment Setup:** Isolation with `venv` (see original README for more details).
*   **Docker Setup:** Containerization for easier deployment (see original README for more details).

**Requirements:**

*   SmokeDetector supports Python versions that are in the [supported phase of the Python life cycle](https://devguide.python.org/versions/)
*   Stack Exchange Login
*   Git 1.8+ (2.11+ recommended) for committing blacklist and watchlist modifications.

**Requesting Removal from Blacklist:**

If you are an official representative of a website and would like to request removal from the blacklist, please see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for details.

**License:**

Licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) or the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>) at your option.

**Contribution Licensing:**

By submitting your contribution for inclusion in the work as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0), you agree that it be dual licensed as above, without any additional terms or conditions.