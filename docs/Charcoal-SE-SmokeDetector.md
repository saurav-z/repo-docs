# SmokeDetector: Real-Time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot that proactively identifies and flags spam on the Stack Exchange network, helping to maintain the integrity of the platform.**  Check out the original repo here: [https://github.com/Charcoal-SE/SmokeDetector](https://github.com/Charcoal-SE/SmokeDetector)

## Key Features:

*   **Real-time Spam Detection:** Monitors the Stack Exchange "realtime tab" to identify and flag spam in real time.
*   **Chatroom Integration:** Posts detected spam to designated chatrooms for community review and action.
*   **Uses Stack Exchange API:** Leverages the Stack Exchange API to access question and answer data.
*   **Customizable Configuration:** Easily configure SmokeDetector with a simple configuration file.
*   **Flexible Deployment Options:** Supports setup via:
    *   Standard Installation
    *   Virtual Environments
    *   Docker Containers
    *   Docker Compose

## Getting Started

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    ```
2.  **Install Dependencies (Standard):**

    ```bash
    git checkout deploy
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
3.  **Install Dependencies (Virtual Environment):**

    ```bash
    python3 -m venv env
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
4.  **Install Dependencies (Docker):**

    ```bash
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    docker create --name=mysmokedetector smokey:$DATE
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```
5.  **Configure:**

    *   Copy `config.sample` to a file named `config` in the root directory.
    *   Edit the `config` file with your specific settings, including API keys, chat room IDs, etc.
    *   Run `python3 nocrash.py` (or `env/bin/python3 nocrash.py` if using a virtual environment) to start SmokeDetector.

## Documentation & Support

*   **User Documentation:** [Wiki](https://charcoal-se.org/smokey)
*   **Setup Guide:** [Set-Up-and-Run-SmokeDetector](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector)

## Requirements

*   **Stack Exchange Login:** SmokeDetector requires a Stack Exchange login.
*   **Python Version:** Supports the Python versions which are in the [supported phase of the Python life cycle](https://devguide.python.org/versions/) (as defined as between "First release" and "End of life").
*   **Git:** Git 1.8 or higher (Git 2.11+ recommended) is required for blacklist/watchlist modifications.

## Blacklist Removal

If you are an official representative of a website seeking removal from the blacklist, please refer to the [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal) for detailed instructions.

## License

SmokeDetector is licensed under either the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) or the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>) at your option.  Contributions are licensed under the same dual-licensing terms.