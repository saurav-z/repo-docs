# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Protect your Stack Exchange communities from spam with SmokeDetector, a headless chatbot that proactively identifies and reports malicious content.**  Learn more and contribute at the [original repository](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector is a powerful tool designed to combat spam on the Stack Exchange network. It operates as a headless chatbot, continuously monitoring the real-time feed and leveraging the Stack Exchange API to detect and report suspicious content to chatrooms.

Key Features:

*   **Real-time Spam Detection:** Monitors the Stack Exchange real-time feed for potential spam.
*   **Automated Reporting:** Posts spam reports to designated chatrooms for community awareness and action.
*   **ChatExchange Integration:** Utilizes the ChatExchange library for seamless chatbot functionality.
*   **Stack Exchange API Access:** Leverages the Stack Exchange API for data retrieval and analysis.
*   **Configurable:**  Easily set up and customize SmokeDetector to fit your community's needs.

Example chat post:

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Setup and Configuration

Detailed setup and running instructions are available in the [wiki](https://charcoal-se.org/smokey). You can choose from different setup methods, including:

*   **Basic Setup:** Clone the repository and install dependencies using `pip3`.
*   **Virtual Environment Setup:** Recommended for isolating dependencies.  Use `venv` to create an isolated environment before installing dependencies.
*   **Docker Setup:** Offers the best isolation. Includes Dockerfile and Docker Compose examples for easy deployment.

## Requirements

*   Stack Exchange login credentials.
*   Python (supports versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/)).
*   Git 1.8+ (recommended 2.11+) for blacklist and watchlist modifications.

## Blacklist Removal

For official representatives of websites seeking blacklist removal, please refer to the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" document.

## License

SmokeDetector is licensed under the following options:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

### Contribution Licensing

By submitting your contribution for inclusion in the work, you agree that it be dual licensed as above, without any additional terms or conditions.