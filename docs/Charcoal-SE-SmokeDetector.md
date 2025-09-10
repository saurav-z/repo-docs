# SmokeDetector: Real-Time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot that instantly identifies and reports spam on the Stack Exchange network, keeping communities clean and helping to protect user experience.** [Learn more on the original GitHub repository](https://github.com/Charcoal-SE/SmokeDetector).

**Key Features:**

*   **Real-time Spam Detection:** Monitors the Stack Exchange realtime tab and identifies spam posts as they appear.
*   **Automated Reporting:** Posts detected spam to designated chatrooms for moderation.
*   **Uses ChatExchange:** Leverages the ChatExchange library for seamless chat integration.
*   **Stack Exchange API Integration:** Accesses question and answer data via the Stack Exchange API.
*   **Flexible Deployment:** Supports setup via direct installation, virtual environments, and Docker containers.

## Getting Started

### Prerequisites

SmokeDetector requires a Stack Exchange login and supports Python versions that are in the [supported phase of the Python life cycle](https://devguide.python.org/versions/). Git 1.8 or higher is also recommended.

## Installation Guides:

### Basic Setup:

1.  Clone the repository:
    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git checkout deploy
    ```
2.  Install dependencies:
    ```bash
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
3.  Configure: Copy `config.sample` to `config` and edit with your settings.
4.  Run: `python3 nocrash.py` (recommended) or `python3 ws.py`.

### Virtual Environment Setup:

1.  Follow steps 1-3 from Basic Setup
2.  Create a virtual environment
    ```bash
    python3 -m venv env
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
3.  Configure: Copy `config.sample` to `config` and edit with your settings.
4.  Run: `env/bin/python3 nocrash.py`.

### Docker Setup:

1.  Build the Docker image:
    ```bash
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  Create a container:
    ```bash
    docker create --name=mysmokedetector smokey:$DATE
    ```
3.  Configure: Copy `config.sample` to `config`, edit, and copy to container:
    ```bash
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```
4.  Start the container: Run the container.

## Docker Compose

If you're familiar with Docker Compose, it's a simple solution for deploying SmokeDetector, by using a `docker-compose.yml` and `config` files.

Run `docker-compose up -d`.

For additional resource control, edit the `docker-compose.yml` and apply resource constraints to the `smokey` service using `mem_limit` and `cpus`.

## Blacklist Removal

If you are an official representative of a blacklisted website/product, please refer to the process for blacklist removal in the wiki: "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)".

## License

SmokeDetector is licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) or the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>), at your option.