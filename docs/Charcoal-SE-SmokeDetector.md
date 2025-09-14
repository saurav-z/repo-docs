# SmokeDetector: Real-time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot that identifies and reports spam on Stack Exchange in real-time, keeping the community clean and efficient.** View the original repo [here](https://github.com/Charcoal-SE/SmokeDetector).

## Key Features:

*   **Real-time Spam Detection:** Monitors Stack Exchange's real-time feed to quickly identify and report spam.
*   **Automated Chat Reporting:** Posts detected spam to designated chatrooms, alerting moderators and users.
*   **Configurable:** Easily set up and customize SmokeDetector to fit your community's specific needs.
*   **Multiple Deployment Options:** Supports setup via basic installation, virtual environments, and Docker for flexible deployment.
*   **Blacklist Management:** Provides tools for managing a blacklist to prevent known spam sites.

## Installation and Setup

Detailed instructions are available in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector) for setting up and running SmokeDetector. Here's a brief overview:

### Basic Setup

1.  **Clone the Repository:** `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  **Navigate and Prepare:** `cd SmokeDetector` and `git checkout deploy`
3.  **Install Dependencies:**  `sudo pip3 install -r requirements.txt --upgrade` and  `pip3 install --user -r user_requirements.txt --upgrade`
4.  **Configure:** Copy `config.sample` to a new file named `config` and edit the necessary values.
5.  **Run:**  `python3 nocrash.py` (recommended for daemon mode) or `python3 ws.py`.

### Virtual Environment Setup

1.  **Clone and Setup:** `git clone https://github.com/Charcoal-SE/SmokeDetector.git`, `cd SmokeDetector`, then set up git config with `git config user.email "smokey@erwaysoftware.com"` and `git config user.name "SmokeDetector"`, followed by `git checkout deploy`.
2.  **Create and Activate Environment:** `python3 -m venv env`.
3.  **Install Dependencies:** `env/bin/pip3 install -r requirements.txt --upgrade` and `env/bin/pip3 install --user -r user_requirements.txt --upgrade`.
4.  **Configure:** Copy `config.sample` to a new file named `config` and edit the necessary values.
5.  **Run:**  `env/bin/python3 nocrash.py`.

### Docker Setup

1.  **Build Docker Image:** Follow the instructions in the documentation and use the provided [Dockerfile](Dockerfile).
2.  **Create a Container:**  Use `docker create` as described in the documentation.
3.  **Start the Container:** Configure the `config` file and copy it into the container using `docker cp`.
4.  **Automated Docker Deployment:** Follow the instructions in the documentation for automated deployment, including details about the `docker-compose.yml` file.

## Requirements

*   SmokeDetector only supports Stack Exchange logins.
*   SmokeDetector supports the Python versions which are in the [supported phase of the Python life cycle](https://devguide.python.org/versions/) (as defined as between "First release" and "End of life"). We run CI testing on that span of versions. SmokeDetector may work on older versions of Python, but we don't support them and may, at any time, write code that prevents use in older, unsupported versions. We know SmokeDetector is broken on Python 3.6 and lower. While we don't support versions that haven't reached "First release", we're not adverse to hearing about changes in new Python versions which will require us to make changes to SmokeDetector's code, so we can make the transition to supporting new versions of Python smoother.
*   Git 1.8 or higher (Git 2.11+ recommended) is required to commit blacklist and watchlist modifications.

## Blacklist Removal

For website removal requests, please review the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for details.

## License

Licensed under Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) and MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>).