# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Tired of spam cluttering your Stack Exchange communities?** SmokeDetector is a powerful, headless chatbot that automatically identifies and reports spam, keeping your chatrooms clean and focused. Explore the original repository on [GitHub](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

## Key Features:

*   **Automated Spam Detection:** Identifies and reports spam in real-time.
*   **Chatroom Integration:** Posts spam reports directly to chatrooms for quick moderation.
*   **Stack Exchange Integration:** Uses the Stack Exchange API and realtime feed to monitor questions.
*   **Flexible Deployment:** Supports various setup methods, including virtual environments and Docker containers.
*   **Configurable:** Easily customize settings via a configuration file.

## Getting Started

Comprehensive documentation and setup instructions are available in the [wiki](https://charcoal-se.org/smokey), with a dedicated section for [setting up and running SmokeDetector](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

### Basic Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate into the directory: `cd SmokeDetector`
3.  Checkout deploy branch: `git checkout deploy`
4.  Install dependencies: `sudo pip3 install -r requirements.txt --upgrade` and `pip3 install --user -r user_requirements.txt --upgrade`
5.  Configure: Copy `config.sample` to `config` and edit your settings.
6.  Run: `python3 nocrash.py` (recommended for continuous operation).

### Virtual Environment Setup

For cleaner dependency management, use a virtual environment:

1.  Follow the same steps as basic setup, but with the following `pip3` command change:
    `env/bin/pip3 install -r requirements.txt --upgrade` and `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
2.  Run:  `env/bin/python3 nocrash.py`

### Docker Setup

Isolate SmokeDetector further with Docker:

1.  Build the Docker image (see original README for instructions, including downloading the Dockerfile).
2.  Create a container from the image.
3.  Start and configure the container with your settings.
4.  (Optional) Use a Bash shell in the container for additional setup.
5.  Automate with Docker Compose (see the original README for instructions).

## Requirements

*   Stack Exchange login credentials.
*   Python (see the original README for a complete explanation, but it's tested on versions currently supported).
*   Git 1.8+ (2.11+ recommended) for contributing back blacklist modifications.

## Blacklist Removal

If you're an official website representative, follow the process for [blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal).

## License

SmokeDetector is dual-licensed under the:

*   [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
*   [MIT license](https://opensource.org/licenses/MIT)

Contributions are also dual-licensed under the same terms.