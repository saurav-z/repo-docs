# SmokeDetector: The Ultimate Spam Detection Bot for Stack Exchange

**Tired of spam cluttering your Stack Exchange chats?** SmokeDetector is a powerful, headless chatbot designed to automatically detect and report spam in real-time, keeping your communities clean and productive.  [Visit the GitHub repository for SmokeDetector](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Circle CI](https://circleci.com/gh/Charcoal-SE/SmokeDetector.svg?style=shield)](https://circleci.com/gh/Charcoal-SE/SmokeDetector)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

## Key Features

*   **Real-time Spam Detection:** Monitors Stack Exchange's real-time feed for spam and malicious content.
*   **Automated Reporting:** Posts detected spam to chatrooms for immediate review and action.
*   **Uses ChatExchange:** Leverages the robust ChatExchange library for seamless chat integration.
*   **API Integration:** Accesses Stack Exchange data via the API.
*   **Customizable:** Easily configured to suit the specific needs of your community.
*   **Multiple Deployment Options:** Supports setup via basic, virtual environment, and Docker.

## Setup and Usage

Detailed documentation and setup instructions are available in the [wiki](https://charcoal-se.org/smokey). Key steps include:

### Basic Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Checkout to deploy branch: `git checkout deploy`
4.  Install dependencies:
    *   `sudo pip3 install -r requirements.txt --upgrade`
    *   `pip3 install --user -r user_requirements.txt --upgrade`
5.  Configure `config`: Copy `config.sample` to a new file named `config` and edit with your settings.
6.  Run the bot: `python3 nocrash.py` (recommended for persistent operation).

### Virtual Environment Setup

1.  Follow steps 1-4 from "Basic Setup"
2.  Create and activate a virtual environment:
    *   `python3 -m venv env`
    *   `source env/bin/activate` (Linux/macOS) or `.\env\Scripts\activate` (Windows)
3.  Configure `config`:  Copy `config.sample` to a new file named `config` and edit with your settings.
4.  Run the bot: `env/bin/python3 nocrash.py`

### Docker Setup

1.  Build the Docker image: (refer to the original readme for detailed steps, these steps are simplified to the most basic use)
    *   `DATE=$(date +%F)`
    *   `docker build -t smokey:$DATE .`
2.  Create a container: `docker create --name=mysmokedetector smokey:$DATE`
3.  Configure and deploy `config` file into container : `docker cp config mysmokedetector:/home/smokey/SmokeDetector/config`
4.  Edit the file `/home/smokey/ready` with the content `touch ~smokey/ready`
5.  Start the container

### Docker Compose Setup

1.  Create a directory, place `config` and `docker-compose.yml` files there.
2.  Run `docker-compose up -d`
3.  For Memory/CPU constraints, edit `docker-compose.yml` as desired.

## Requirements

*   Stack Exchange Login credentials
*   Python 3.7 or higher (supporting versions listed in readme)
*   Git 1.8+ (2.11+ recommended)

## Requesting Blacklist Removal

If you are an official representative of a website/product and wish to request removal from the blacklist, please refer to the [process](https://charcoal-se.org/smokey/Process-for-blacklist-removal).

## License

SmokeDetector is licensed under the Apache License, Version 2.0 or the MIT license. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.