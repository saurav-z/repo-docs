# SmokeDetector: Real-Time Spam Detection for Stack Exchange

**SmokeDetector is a headless chatbot meticulously designed to identify and report spam in real-time across the Stack Exchange network.** [View the original repository on GitHub](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

## Key Features

*   **Real-time Spam Detection:** Monitors Stack Exchange for spam activity.
*   **Chatroom Reporting:**  Posts detected spam to designated chatrooms.
*   **Utilizes Standard APIs:** Leverages the Stack Exchange API and ChatExchange for comprehensive data access.
*   **Flexible Deployment Options:** Supports setup via Git, virtual environments, and Docker containers.
*   **Detailed Documentation:** Comprehensive user and setup guides available.

## Getting Started

*   **Installation:**
    *   Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
    *   Navigate to the directory: `cd SmokeDetector`
    *   Follow the instructions in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector) for setting up and running.

## Setup Guides

Detailed instructions on setting up SmokeDetector are in the [wiki](https://charcoal-se.org/smokey). This includes setup via:

*   [Basic Setup](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector#basic-setup)
*   [Virtual Environment Setup](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector#virtual-environment-setup)
*   [Docker Setup](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector#docker-setup)
*   [Docker Compose Setup](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector#automate-docker-deployment-with-docker-compose)

## Requirements

*   **Python:** Supports Python versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/).
*   **Git:**  Git 1.8 or higher (2.11+ recommended).
*   **Stack Exchange Login:** SmokeDetector supports Stack Exchange logins.

## Blacklist Removal

For website/product removal requests, see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)".

## License

SmokeDetector is licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT license](LICENSE-MIT).