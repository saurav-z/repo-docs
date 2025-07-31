# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Tired of spam cluttering your Stack Exchange chats? SmokeDetector is a powerful, open-source chatbot designed to identify and report spam in real-time.** [Learn more and contribute on GitHub](https://github.com/Charcoal-SE/SmokeDetector).

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

**Key Features:**

*   **Real-time Spam Detection:** Identifies and reports spam quickly using data from the Stack Exchange realtime tab and API.
*   **Headless Chatbot:** Operates in the background, automatically posting spam reports to designated chatrooms.
*   **Easy to Set Up:** Detailed setup instructions are provided in the wiki (link below) for various installation methods, including virtual environments and Docker.
*   **Open Source:** Available under the Apache License, Version 2.0 and MIT license, allowing for community contributions and customization.
*   **Flexible Deployment:** Supports various deployment methods, including standard installation, virtual environments, and Docker containers.
*   **Blacklist Management:** Includes functionality for blacklisting websites and a process for requesting removal.

**How it Works:**

SmokeDetector utilizes [ChatExchange](https://github.com/Manishearth/ChatExchange) to connect to chatrooms, monitors the Stack Exchange [realtime tab](https://stackexchange.com/questions?tab=realtime) for new questions, and uses the [Stack Exchange API](https://api.stackexchange.com/) to access answer content.

**Documentation:**

*   **User Documentation:** [Wiki](https://charcoal-se.org/smokey)
*   **Setup and Running:** [Wiki Setup Instructions](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector)
*   **Example chat post:** ![Example chat post](https://i.sstatic.net/oLyfb.png)

**Installation and Setup:**

Choose your preferred setup method:

1.  **Basic Installation:**
    ```shell
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git checkout deploy
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
    Copy `config.sample` to `config` and edit the required values. Run using `python3 nocrash.py`.

2.  **Virtual Environment Setup:**
    ```shell
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git config user.email "smokey@erwaysoftware.com"
    git config user.name "SmokeDetector"
    git checkout deploy

    python3 -m venv env
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
    Copy `config.sample` to `config` and configure. Run using `env/bin/python3 nocrash.py`.

3.  **Docker Setup:**
    Follow the Docker setup instructions in the original README.

**Requirements:**

*   Stack Exchange login
*   Python versions within the [supported phase of the Python life cycle](https://devguide.python.org/versions/)
*   Git 1.8 or higher (Git 2.11+ recommended)

**Requesting Blacklist Removal:**

If you are an official representative of a website and wish to request removal from the blacklist, please see the [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal).

**License:**

Licensed under either of:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

**Contribution Licensing:**

By contributing, you agree to dual license your contributions as stated above.