# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Protecting the Stack Exchange community from spam, SmokeDetector is a powerful, headless chatbot that identifies and reports malicious content in real-time.** (Original repository: [Charcoal-SE/SmokeDetector](https://github.com/Charcoal-SE/SmokeDetector))

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

SmokeDetector is a headless chatbot designed to detect and report spam on the Stack Exchange network. It monitors the real-time activity of Stack Exchange, analyzes content, and posts suspicious items to chat rooms for review.

## Key Features:

*   **Real-time Spam Detection:** Continuously monitors Stack Exchange for potentially malicious content.
*   **Chat Integration:** Posts suspected spam to chat rooms for community review.
*   **Uses ChatExchange:** Leverages the ChatExchange library for communication.
*   **API Integration:** Accesses questions and answers using the Stack Exchange API.
*   **Flexible Deployment:** Supports setup with standard, virtual environment, and Docker.

Example [chat post](https://chat.stackexchange.com/transcript/message/43579469):

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Getting Started

### Setup and Installation

You can set up SmokeDetector using several methods: standard, virtual environment, and Docker. Detailed setup instructions are in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

**Basic Setup:**

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Checkout the deploy branch: `git checkout deploy`
4.  Install dependencies: `sudo pip3 install -r requirements.txt --upgrade` and `pip3 install --user -r user_requirements.txt --upgrade`
5.  Configure: Copy `config.sample` to `config` and edit the values.
6.  Run:  `python3 nocrash.py` (recommended) or `python3 ws.py`

**Virtual Environment Setup:**

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Configure Git: `git config user.email "smokey@erwaysoftware.com"` and `git config user.name "SmokeDetector"`
4.  Checkout the deploy branch: `git checkout deploy`
5.  Create and activate the environment: `python3 -m venv env` and `env/bin/pip3 install -r requirements.txt --upgrade` and `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
6.  Configure: Copy `config.sample` to `config` and edit the values.
7.  Run:  `env/bin/python3 nocrash.py`

**Docker Setup:**

1.  Build the Docker image.
2.  Create a container.
3.  Start the container and configure the `config` file.
4.  (Optional) Set up additional tools within the container.

  *   You can automate Docker deployment with Docker Compose.

## Requirements

*   Stack Exchange Login.
*   Supported Python versions (between "First release" and "End of life").
*   Git 1.8 or higher (Git 2.11+ recommended).

## Blacklist Removal

Official representatives of websites seeking blacklist removal should follow the process outlined in the [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal).

## License

SmokeDetector is available under the following licenses:

*   Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

### Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.