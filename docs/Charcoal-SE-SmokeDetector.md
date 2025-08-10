# SmokeDetector: Real-time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot that instantly identifies and reports spam on Stack Exchange, protecting communities from unwanted content.** [View the original repo.](https://github.com/Charcoal-SE/SmokeDetector)

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector utilizes the Stack Exchange API and the realtime tab to scan questions, identifies spam, and posts reports to designated chatrooms.

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Key Features:

*   **Real-time Spam Detection:** Quickly identifies and flags spam on Stack Exchange.
*   **Automated Reporting:** Posts spam reports directly to chatrooms, enabling moderators and community members to take action.
*   **Headless Operation:** Runs in the background as a chatbot, without requiring manual monitoring.
*   **Flexible Setup:** Supports multiple deployment methods, including direct installation, virtual environments, and Docker containers.
*   **Community-Driven:** Contribute to the project with your suggestions and changes!

## Getting Started

Detailed setup and running instructions are available in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector). Here's a quick overview of the installation methods:

### Basic Setup
1.  Clone the repository
2.  Navigate into the directory.
3.  Checkout the deploy branch
4.  Install requirements using pip.
5.  Configure by copying `config.sample` to `config` and editing values.
6.  Run the program using `python3 nocrash.py`.

### Virtual Environment Setup
1.  Clone the repository
2.  Navigate into the directory.
3.  Git configure the user email and name
4.  Checkout the deploy branch
5.  Create a virtual environment.
6.  Install requirements using pip.
7.  Configure by copying `config.sample` to `config` and editing values.
8.  Run the program using `env/bin/python3 nocrash.py`.

### Docker Setup
1.  Get the [Dockerfile](Dockerfile)
2.  Build the image.
3.  Create a container.
4.  Edit config and copy it to the container.
5.  Run the container and set the ready file.

### Docker Compose Setup
1.  Have a properly filled `config` file.
2.  Create a directory, place the `config` file and [`docker-compose.yml` file](docker-compose.yml).
3.  Run `docker-compose up -d`

## Requirements

SmokeDetector supports Python versions as described in the [supported phase of the Python life cycle](https://devguide.python.org/versions/).

## Blacklist Removal

If you are a website representative wanting removal from the blacklist, please see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)".

## License

This project is licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) or the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>).