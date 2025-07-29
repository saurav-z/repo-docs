# SmokeDetector: Real-Time Spam Detection for Stack Exchange

Tired of spam cluttering your Stack Exchange communities? SmokeDetector is a powerful, headless chatbot that identifies and reports spam in real-time.  [Check out the original repository](https://github.com/Charcoal-SE/SmokeDetector) for more details.

## Key Features:

*   **Real-time Spam Detection:** Monitors the Stack Exchange realtime feed to identify potentially malicious content.
*   **Automated Reporting:** Posts detected spam to designated chatrooms for immediate attention.
*   **ChatExchange Integration:** Leverages the ChatExchange library for seamless communication within Stack Exchange chatrooms.
*   **API Driven:** Uses the Stack Exchange API to access questions and answers.
*   **Flexible Setup:** Offers multiple setup options, including virtual environments and Docker containers, for easy deployment.

## Getting Started

Choose your preferred setup method:

### Basic Setup
1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Checkout the deploy branch: `git checkout deploy`
4.  Install dependencies: `sudo pip3 install -r requirements.txt --upgrade`
5.  Install user requirements: `pip3 install --user -r user_requirements.txt --upgrade`
6.  Configure: Copy `config.sample` to `config` and edit values.
7.  Run: `python3 nocrash.py` (recommended for daemon-like operation) or `python3 ws.py`.

### Virtual Environment Setup
1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Set up your git user info: `git config user.email "smokey@erwaysoftware.com"` and `git config user.name "SmokeDetector"`
4.  Checkout the deploy branch: `git checkout deploy`
5.  Create and activate a virtual environment:  `python3 -m venv env`
6.  Install dependencies: `env/bin/pip3 install -r requirements.txt --upgrade`
7.  Install user requirements: `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
8.  Configure: Copy `config.sample` to `config` and edit values.
9.  Run: `env/bin/python3 nocrash.py`

### Docker Setup
1.  Grab the [Dockerfile](Dockerfile).
2.  Build the image:
    ```shell
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
3.  Create a container: `docker create --name=mysmokedetector smokey:$DATE`
4.  Start the container.
5.  Configure: Copy `config.sample` to `config` inside the container:
    ```shell
    docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
    ```
6.  (Optional) Access the container's bash shell for additional setup: `docker exec -it mysmokedetector bash` and add `/home/smokey/ready`
7.  Automate with Docker Compose (see [docker-compose.yml](docker-compose.yml) for more).

## Requirements

*   **Stack Exchange Login:** SmokeDetector requires a Stack Exchange login.
*   **Python:** Supports Python versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/) (between "First release" and "End of life"). Python 3.7+ is recommended.
*   **Git:** Git 1.8+ is required for committing blacklist and watchlist modifications (2.11+ recommended).

## Documentation and Resources

*   **User Documentation:** [Wiki](https://charcoal-se.org/smokey)
*   **Setup & Run:** [Wiki - Set-Up-and-Run-SmokeDetector](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector)
*   **Blacklist Removal:** [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)

## License

SmokeDetector is available under the terms of both the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) and the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>), at your option.