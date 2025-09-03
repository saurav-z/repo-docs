# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Tired of spam cluttering your Stack Exchange chatrooms? SmokeDetector is your solution!** This headless chatbot actively monitors Stack Exchange for spam, automatically posting detections to chatrooms for quick moderation.  Learn more and contribute at the [original repository](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

## Key Features

*   **Real-time Spam Detection:** Identifies and flags spam posts on Stack Exchange.
*   **Automated Chat Posting:**  Posts spam detections to designated chatrooms for moderation.
*   **Uses Stack Exchange APIs:** Leverages the Stack Exchange API to access data and monitor activity.
*   **Headless Chatbot:** Operates in the background, requiring no user interaction.
*   **Flexible Deployment:** Supports various setups including: standard, virtual environment, and Docker.

## How it Works

SmokeDetector uses [ChatExchange](https://github.com/Manishearth/ChatExchange) to interact with Stack Exchange chatrooms. It monitors the Stack Exchange [realtime tab](https://stackexchange.com/questions?tab=realtime) for new questions and uses the [Stack Exchange API](https://api.stackexchange.com/) to gather information.

**Example:**

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Getting Started

### Installation

1.  **Clone the repository:** `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  **Navigate to the directory:** `cd SmokeDetector`
3.  **Checkout the deploy branch:** `git checkout deploy`
4.  **Install requirements:** `sudo pip3 install -r requirements.txt --upgrade && pip3 install --user -r user_requirements.txt --upgrade`
5.  **Configure:** Copy `config.sample` to a file named `config` and edit with your credentials and settings.
6.  **Run:** `python3 nocrash.py` (recommended for continuous operation) or `python3 ws.py` (will shut down after 6 hours)

### Additional Setup Options

*   **Virtual Environment Setup:**  Isolate dependencies using Python's `venv` module.
*   **Docker Setup:** Provides the most isolated setup via containerization, with included `Dockerfile` and `docker-compose.yml` examples.

**See the detailed setup guides in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).**

## Requirements

*   Stack Exchange login credentials.
*   Supported Python versions within the [supported phase of the Python life cycle](https://devguide.python.org/versions/) (as defined as between "First release" and "End of life").
*   Git 1.8+ (2.11+ recommended) for committing blacklist and watchlist modifications.

## Blacklist Removal

If you are an official representative of a website wishing to be removed from the blacklist, please see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for details.

## License

This project is licensed under the following options:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

### Contribution Licensing

By submitting your contribution for inclusion in the work, you agree that it be dual licensed as above, without any additional terms or conditions.