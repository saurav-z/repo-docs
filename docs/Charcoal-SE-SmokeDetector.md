# SmokeDetector: Real-Time Spam Detection for Stack Exchange

Tired of spam on Stack Exchange? SmokeDetector is a powerful, headless chatbot that automatically identifies and reports spam in real-time, keeping your communities clean and safe. Learn more on the original repository: [https://github.com/Charcoal-SE/SmokeDetector](https://github.com/Charcoal-SE/SmokeDetector)

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

**Key Features:**

*   **Real-time Spam Detection:** Identifies spam questions and posts instantly.
*   **Automated Reporting:** Posts detected spam to chatrooms for community review.
*   **Stack Exchange Integration:** Leverages the Stack Exchange realtime tab and API for efficient monitoring.
*   **Headless Chatbot:** Runs in the background, minimizing impact on community.
*   **Customizable and Extensible:** Adaptable to various Stack Exchange communities.

Example chat post:

![Example chat post](https://i.sstatic.net/oLyfb.png)

## Getting Started

### Basic Setup

1.  **Clone the Repository:** `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  **Navigate:** `cd SmokeDetector`
3.  **Checkout:** `git checkout deploy`
4.  **Install Dependencies:**  `sudo pip3 install -r requirements.txt --upgrade` followed by `pip3 install --user -r user_requirements.txt --upgrade`
5.  **Configure:** Copy `config.sample` to `config` and edit the values.
6.  **Run:** `python3 nocrash.py` (recommended for continuous operation) or `python3 ws.py` (will shut down after 6 hours)

### Virtual Environment Setup

Using a virtual environment helps isolate dependencies.

1.  **Clone the Repository:** `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  **Navigate and Configure Git:** `cd SmokeDetector`, then set `git config user.email` and `git config user.name`.
3.  **Checkout:** `git checkout deploy`
4.  **Create and Activate Environment:** `python3 -m venv env`
5.  **Install Dependencies:** `env/bin/pip3 install -r requirements.txt --upgrade` followed by `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
6.  **Configure:** Copy `config.sample` to `config` and edit the values.
7.  **Run:**  `env/bin/python3 nocrash.py`

### Docker Setup

Utilize Docker for a containerized environment.

1.  **Build the Image:** Follow the instructions using `Dockerfile`.
2.  **Create a Container:**  Use the `docker create` command.
3.  **Start the Container:** Configure your `config` file and copy it into the container.
4.  **Additional Configuration (optional):**  Access the container's bash shell via `docker exec -it`.
5.  **Signal Ready:**  Create the `ready` file.

#### Automate Docker Deployment with Docker Compose

1.  **Prepare `config`:** Create a valid `config` file.
2.  **Create `docker-compose.yml`:** Place it in the same directory.
3.  **Run:** Execute `docker-compose up -d`.
4.  **Customize (optional):**  Modify `docker-compose.yml` to set resource limits.

## Documentation

Refer to the [wiki](https://charcoal-se.org/smokey) for comprehensive user and setup documentation, including:
*   Detailed instructions for setting up and running SmokeDetector.

## Requirements

*   Stack Exchange Login
*   Python versions within the [supported phase](https://devguide.python.org/versions/) (as defined as between "First release" and "End of life") of the Python life cycle
*   Git 1.8 or higher (2.11+ recommended) for blacklist and watchlist modifications.

## Blacklist Removal Requests

If you are an official representative seeking to remove a website from the blacklist, consult the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for detailed instructions.

## License

SmokeDetector is available under the following licenses:

*   Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

## Contribution Licensing

Your contributions are subject to the dual licensing terms as outlined in the Apache-2.0 license.