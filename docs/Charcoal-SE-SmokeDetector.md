# SmokeDetector: The Ultimate Spam Detection Chatbot for Stack Exchange

Detect and flag spam on Stack Exchange with SmokeDetector, a powerful and versatile chatbot. Find the original repository [here](https://github.com/Charcoal-SE/SmokeDetector).

SmokeDetector monitors Stack Exchange in real-time, identifies spam, and posts alerts to designated chatrooms.

## Key Features:

*   **Real-time Spam Detection:** Leverages Stack Exchange's real-time feed to identify spam quickly.
*   **Automated Chatroom Posting:**  Posts identified spam to chatrooms for review and moderation.
*   **Customizable Configuration:**  Easy to configure with a `config` file to match your needs.
*   **Multiple Setup Options:** Offers setup via standard installation, virtual environments, and Docker containers.
*   **Blacklist Integration:** Supports blacklists to efficiently filter known spam sources.

## Getting Started

### Installation

Choose your preferred method:

*   **Standard Setup:**
    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git checkout deploy
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
*   **Virtual Environment Setup:**

    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git config user.email "smokey@erwaysoftware.com"
    git config user.name "SmokeDetector"
    git checkout deploy

    python3 -m venv env
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```

*   **Docker Setup:** Follow the Docker instructions detailed in the documentation.

### Configuration

1.  Copy `config.sample` and rename it to `config`.
2.  Edit the `config` file with your desired settings.
3.  Run SmokeDetector using `python3 nocrash.py` (recommended) or `python3 ws.py` (shuts down after 6 hours). For virtual environments, run with `env/bin/python3 nocrash.py`.

## Deployment with Docker Compose (Recommended)

1.  Ensure your `config` file is correctly filled out.
2.  Create a directory, place the `config` file and `docker-compose.yml` file.
3.  Run `docker-compose up -d` to start SmokeDetector.
4.  You can customize resource limits within `docker-compose.yml`:

    ```yaml
    restart: always
    mem_limit: 512M
    cpus: 0.5
    ```

## Requirements

*   **Python:** Supports Python versions within the supported phase of the Python life cycle.
*   **Git:** Git 1.8 or higher (2.11+ recommended) is required for blacklist/watchlist modifications.
*   **Stack Exchange Login:** SmokeDetector only supports Stack Exchange logins.

## Documentation

For detailed instructions and advanced usage, please see the comprehensive documentation on the [wiki](https://charcoal-se.org/smokey).

*   [Setting up and running SmokeDetector](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector)

## Blacklist Removal

If you are an official representative of a website seeking removal from the blacklist, please see the [blacklist removal process](https://charcoal-se.org/smokey/Process-for-blacklist-removal) for detailed instructions.

## License

SmokeDetector is licensed under either the:

*   [Apache License, Version 2.0](LICENSE-APACHE)
*   [MIT license](LICENSE-MIT)

at your option.