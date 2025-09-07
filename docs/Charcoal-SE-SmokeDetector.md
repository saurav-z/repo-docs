# SmokeDetector: Real-Time Spam Detection for Stack Exchange

**Tired of spam cluttering your Stack Exchange communities? SmokeDetector is a powerful, headless chatbot that automatically identifies and reports spam in real-time.** [Learn more and contribute on GitHub!](https://github.com/Charcoal-SE/SmokeDetector)

## Key Features:

*   **Real-time Spam Detection:** Monitors Stack Exchange's realtime tab for new questions and answers.
*   **Automated Reporting:** Posts suspected spam to designated chatrooms for review by moderators and community members.
*   **Stack Exchange API Integration:** Leverages the Stack Exchange API to access question and answer details.
*   **Configurable:** Easily set up and customize SmokeDetector to meet your community's specific needs.

## Setup and Installation:

Choose your preferred setup method:

*   **Basic Setup:**

    1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
    2.  Navigate to the directory: `cd SmokeDetector`
    3.  Switch to the deploy branch: `git checkout deploy`
    4.  Install dependencies: `sudo pip3 install -r requirements.txt --upgrade` and `pip3 install --user -r user_requirements.txt --upgrade`
    5.  Configure: Copy `config.sample` to `config` and edit with your settings.
    6.  Run: `python3 nocrash.py` (recommended for continuous operation) or `python3 ws.py`
*   **Virtual Environment Setup:**

    1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
    2.  Navigate to the directory: `cd SmokeDetector`
    3.  Configure Git settings: `git config user.email "smokey@erwaysoftware.com"` and `git config user.name "SmokeDetector"`
    4.  Switch to the deploy branch: `git checkout deploy`
    5.  Create and activate the virtual environment: `python3 -m venv env` and `env/bin/pip3 install -r requirements.txt --upgrade` and `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
    6.  Configure: Copy `config.sample` to `config` and edit with your settings.
    7.  Run: `env/bin/python3 nocrash.py`
*   **Docker Setup:**

    1.  **Build the Docker image:**
        ```bash
        DATE=$(date +%F)
        mkdir temp
        cd temp
        wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
        docker build -t smokey:$DATE .
        ```
    2.  **Create a container:** `docker create --name=mysmokedetector smokey:$DATE`
    3.  **Configure and run:**
        *   Copy `config.sample` to `config` and edit with your settings.
        *   Copy the config file into the container: `docker cp config mysmokedetector:/home/smokey/SmokeDetector/config`
        *   **Start container:** (The smokey won't run until the config is ready)
    4.  **Advanced configuration inside Docker container:**  `docker exec -it mysmokedetector bash`, `touch ~smokey/ready`
    5.  **Automate Deployment with Docker Compose:**
        *   Create a `config` file based on `config.sample`
        *   Create a `docker-compose.yml` file
        *   Run `docker-compose up -d`

    *   **Docker Compose additional configuration**
        ```yaml
        restart: always  # when your host reboots Smokey can autostart
        mem_limit: 512M
        cpus: 0.5  # Recommend 2.0 or more for spam waves
        ```
## Requirements:

*   Python (See supported Python versions in the wiki)
*   Git (1.8+ recommended, 2.11+ for blacklist/watchlist modifications)
*   Stack Exchange login

## Documentation:

*   **User Documentation:** [Wiki](https://charcoal-se.org/smokey)
*   **Setup & Running:** [Setting Up and Running SmokeDetector](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector)

## Blacklist Removal:

Official representatives of websites can request removal from the blacklist: "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)"

## License:

Licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) or the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>) at your option.  Contributions are dual-licensed under the same terms.