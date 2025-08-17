# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Tired of spam flooding your Stack Exchange communities? SmokeDetector is a powerful, headless chatbot that automatically identifies and reports spam in real-time.**

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector leverages the Stack Exchange API and the realtime tab to monitor for and flag unwanted content.

[View the original repository on GitHub](https://github.com/Charcoal-SE/SmokeDetector)

## Key Features:

*   **Real-time Spam Detection:** Monitors Stack Exchange for spam and malicious content.
*   **Automated Reporting:** Posts detected spam to chatrooms for community review.
*   **Open-Source:** Freely available and customizable to meet your community's needs.
*   **Easy to Deploy:** Supports setup via Git, virtual environments, and Docker.

## Documentation & Setup

Comprehensive user and setup documentation is available on the [wiki](https://charcoal-se.org/smokey), including details for setting up and running SmokeDetector.

### Installation Guide

Choose your preferred installation method:

*   **Basic Setup (Git):** Clone the repository and install dependencies using `pip3`.
*   **Virtual Environment:** Create an isolated environment to manage dependencies.
*   **Docker:** Build and run SmokeDetector within a Docker container for easy deployment and isolation.

## Requirements

*   Stack Exchange login credentials
*   Git 1.8+ (recommended: 2.11+) for contributing blacklist and watchlist modifications.
*   Python versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/).

## Blacklist Removal

Official website representatives can request removal from the blacklist. See "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for details.

## License

SmokeDetector is licensed under either:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)