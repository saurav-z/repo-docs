# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Instantly identify and report spam on Stack Exchange using SmokeDetector, a powerful and customizable chatbot.** ([View the original repository](https://github.com/Charcoal-SE/SmokeDetector))

SmokeDetector is a headless chatbot that monitors the Stack Exchange network in real-time, identifying and reporting spam posts to designated chatrooms. Built using [ChatExchange](https://github.com/Manishearth/ChatExchange) and the Stack Exchange API, SmokeDetector provides a crucial service in maintaining the integrity of the Stack Exchange platform.

**Key Features:**

*   **Real-time Monitoring:**  Tracks the Stack Exchange real-time feed for potential spam.
*   **Chatroom Reporting:** Posts detected spam to chatrooms for community review and action.
*   **Automated Detection:** Employs sophisticated logic to identify spam based on various criteria.
*   **Customizable:** Easy to set up and configure to meet specific needs.
*   **Multiple Deployment Options:** Supports setup via standard installation, virtual environments, and Docker containers.

## Getting Started

Detailed setup and running instructions are available in the [wiki](https://charcoal-se.org/smokey/).

### Basic Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git checkout deploy
    ```

2.  **Install dependencies:**

    ```bash
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```

3.  **Configure:**
    *   Copy `config.sample` to a new file named `config`.
    *   Edit the values in `config` with your specific credentials and settings.
4.  **Run:**
    *   `python3 nocrash.py` (recommended for continuous operation)
    *   Alternatively, `python3 ws.py` (will shut down after 6 hours)

### Virtual Environment Setup

To isolate dependencies, use a virtual environment:

1.  **Follow steps 1 & 2 from Basic Setup**
2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv env
    ```

3.  **Install dependencies (within the virtual environment):**

    ```bash
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```

4.  **Configure**
    *   Copy `config.sample` to a new file named `config`.
    *   Edit the values in `config` with your specific credentials and settings.
5.  **Run (within the virtual environment):**

    ```bash
    env/bin/python3 nocrash.py
    ```

### Docker Setup

For enhanced isolation, use Docker:

1.  **Build the Docker image:**

    ```bash
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```

2.  **Create a container:**

    ```bash
    docker create --name=mysmokedetector smokey:$DATE
    ```

3.  **Configure:**

    *   Copy `config.sample` to `config`.
    *   Edit the values in `config` with your specific credentials and settings.
    *   Copy the `config` file into the container:
        ```bash
        docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
        ```

4.  **Optional: Access the container shell:**

    ```bash
    docker exec -it mysmokedetector bash
    ```

    Create a `ready` file in `/home/smokey` after configuration is complete:

    ```bash
    touch ~smokey/ready
    ```

### Docker Compose

For automated Docker deployment, you can use Docker Compose

1.  **Create a directory and place `config` and `docker-compose.yml` in it**.
2.  **Run:**
    `docker-compose up -d`

*For additional control (e.g. for memory, CPU), edit `docker-compose.yml`*

## Requirements

*   Stack Exchange login.
*   Python versions that are in the [supported phase of the Python life cycle](https://devguide.python.org/versions/) (as defined as between "First release" and "End of life").
*   Git 1.8+ (2.11+ recommended) to commit blacklist/watchlist modifications.

## Blacklist Removal

For website/product removal requests, see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)".

## License

This project is dual-licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) and the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>).

### Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.