# SmokeDetector: Real-time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot designed to automatically detect and report spam on the Stack Exchange network, protecting the community from unwanted content.** Learn more and contribute on the [original repository](https://github.com/Charcoal-SE/SmokeDetector).

## Key Features:

*   **Real-time Spam Detection:** Monitors Stack Exchange's real-time feed for suspicious content.
*   **Automated Reporting:** Posts detected spam to designated chatrooms for community review.
*   **ChatExchange Integration:** Utilizes the ChatExchange library to interact with Stack Exchange chat.
*   **API Integration:** Leverages the Stack Exchange API to access question and answer data.
*   **Flexible Deployment:** Supports various deployment methods, including direct setup, virtual environments, and Docker containers.

## Getting Started

### Prerequisites

*   Stack Exchange login credentials.
*   Python (supported versions are listed in the Requirements section).
*   Git 1.8 or higher (recommended 2.11+).

### Installation

Choose your preferred setup method:

#### Basic Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Switch to the deployment branch: `git checkout deploy`
4.  Install dependencies:
    ```shell
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
5.  Configure: Copy `config.sample` to a new file named `config` and edit with your settings.
6.  Run: `python3 nocrash.py` (recommended for continuous operation) or `python3 ws.py` (shuts down after 6 hours).

#### Virtual Environment Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Configure git:
    ```shell
    git config user.email "smokey@erwaysoftware.com"
    git config user.name "SmokeDetector"
    ```
4.  Switch to the deployment branch: `git checkout deploy`
5.  Create and activate a virtual environment:
    ```shell
    python3 -m venv env
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
6.  Configure: Copy `config.sample` to a new file named `config` and edit with your settings.
7.  Run: `env/bin/python3 nocrash.py`

#### Docker Setup

1.  **Build the Docker image:** Follow the steps in the original README.
2.  **Create and Start Container:** Follow the steps in the original README.
3.  **Configure and Run:** Follow the steps in the original README.

#### Automate Docker deployment with Docker Compose

1.  Create a directory, and place `config` file and [`docker-compose.yml` file](docker-compose.yml).
2.  Run `docker-compose up -d`

## Requirements

SmokeDetector supports Python versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/). Git 1.8 or higher (recommended 2.11+) is required for blacklist/watchlist modifications.

## Additional Resources

*   **User Documentation:** [Wiki](https://charcoal-se.org/smokey)
*   **Process for Blacklist Removal:** [Blacklist Removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)

## License

SmokeDetector is licensed under either the Apache License, Version 2.0, or the MIT license, at your option. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.