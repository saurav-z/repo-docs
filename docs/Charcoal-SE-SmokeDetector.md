# SmokeDetector: Real-time Spam Detection for Stack Exchange

**SmokeDetector** is a powerful, headless chatbot designed to identify and report spam across the Stack Exchange network.  Check out the [original repository](https://github.com/Charcoal-SE/SmokeDetector) for more details.

## Key Features:

*   **Real-time Spam Detection:** Monitors Stack Exchange's real-time feed to identify potentially malicious content.
*   **Automated Reporting:** Posts detected spam to designated chatrooms for community review and action.
*   **Headless Operation:** Runs autonomously, requiring minimal manual intervention.
*   **Uses Stack Exchange APIs:** Leverages the Stack Exchange API and ChatExchange for data access and chat integration.
*   **Flexible Setup:** Supports various setup methods, including standard, virtual environment, and Docker for optimized deployment.

## Getting Started:

### Basic Setup:

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Switch to the deploy branch: `git checkout deploy`
4.  Install dependencies: `sudo pip3 install -r requirements.txt --upgrade`
5.  Install user requirements: `pip3 install --user -r user_requirements.txt --upgrade`
6.  Configure: Copy `config.sample` to `config` and fill in the required values.
7.  Run:  `python3 nocrash.py` (recommended) or `python3 ws.py`

### Virtual Environment Setup

1. Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2. Navigate to the directory: `cd SmokeDetector`
3. Configure user details:
    `git config user.email "smokey@erwaysoftware.com"`
    `git config user.name "SmokeDetector"`
4. Switch to the deploy branch: `git checkout deploy`
5. Create and activate the environment: `python3 -m venv env`
6. Install dependencies: `env/bin/pip3 install -r requirements.txt --upgrade`
7. Install user requirements: `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
8. Configure: Copy `config.sample` to `config` and fill in the required values.
9. Run: `env/bin/python3 nocrash.py`

### Docker Setup:

1.  **Build the Docker Image:**  Follow the instructions in the original README for building and setting up a container.
2.  **Configure:** Copy `config.sample` to `config` and populate it with the correct information.
3.  **Run:** Follow the Docker deployment instructions to start SmokeDetector within your container.

## Requirements:

*   Stack Exchange login credentials.
*   Python 3.7 or higher (matching the Python versions under active development).
*   Git 1.8 or higher (Git 2.11+ recommended) for blacklist and watchlist modifications.

## Blacklist Removal Requests:

If you are an official representative of a website and wish to request removal from the blacklist, please see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for details.

## License:

SmokeDetector is available under the Apache License, Version 2.0, or the MIT license at your option. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for more details.