# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Tired of spam polluting your Stack Exchange communities? SmokeDetector is a headless chatbot that automatically identifies and reports spam in real-time.**

[View the original repository on GitHub](https://github.com/Charcoal-SE/SmokeDetector)

SmokeDetector analyzes Stack Exchange questions and answers using the Stack Exchange API and posts spam reports to chatrooms, helping moderators and users quickly address malicious content.

**Key Features:**

*   **Real-time Spam Detection:** Identifies spam as it appears on Stack Exchange.
*   **Automated Reporting:** Posts identified spam to chatrooms for review.
*   **Utilizes Stack Exchange API:** Leverages the API for access to question and answer data.
*   **Flexible Deployment:** Supports various deployment methods, including virtual environments and Docker containers.
*   **Customizable:** Configuration options to tailor detection and reporting behavior.

**Getting Started:**

SmokeDetector provides several setup options:

*   **Basic Setup:** A simple way to get SmokeDetector running on your system.
    *   Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
    *   Navigate to the directory: `cd SmokeDetector`
    *   Switch to the deploy branch: `git checkout deploy`
    *   Install dependencies: `sudo pip3 install -r requirements.txt --upgrade && pip3 install --user -r user_requirements.txt --upgrade`
    *   Configure the application: Copy `config.sample` to a new file named `config` and edit the necessary values.
    *   Run the application: `python3 nocrash.py` (recommended) or `python3 ws.py`.
*   **Virtual Environment Setup:** Isolates dependencies for cleaner installations.
    *   Clone the repository and configure git user settings:
       `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
       `cd SmokeDetector`
       `git config user.email "smokey@erwaysoftware.com"`
       `git config user.name "SmokeDetector"`
       `git checkout deploy`
    *   Set up the virtual environment:
       `python3 -m venv env`
       `env/bin/pip3 install -r requirements.txt --upgrade`
       `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
    *   Configure the application: Copy `config.sample` to a new file named `config` and edit the necessary values.
    *   Run the application: `env/bin/python3 nocrash.py`.
*   **Docker Setup:** Simplifies dependency management and provides a containerized environment. (See the Dockerfile for setup details)
*   **Docker Compose:** Automates Docker deployment for a streamlined experience. (See docker-compose.yml file and instructions)

**Requirements:**

*   Stack Exchange Login
*   Git 1.8+ (2.11+ recommended)
*   Python 3.7+

**Documentation:**

*   User documentation is available on the [wiki](https://charcoal-se.org/smokey).
*   Detailed setup instructions are found in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

**Blacklist Removal:**
Official representatives of websites/products can request removal from the blacklist following the process outlined [here](https://charcoal-se.org/smokey/Process-for-blacklist-removal).

**License:**

SmokeDetector is licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) or the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>), at your option.