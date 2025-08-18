# SmokeDetector: Real-time Spam Detection for Stack Exchange

SmokeDetector is a powerful, headless chatbot designed to identify and report spam on the Stack Exchange network in real-time. Visit the [original repository](https://github.com/Charcoal-SE/SmokeDetector) to learn more.

## Key Features:

*   **Automated Spam Detection:** Analyzes Stack Exchange questions in real-time to identify and flag potential spam.
*   **Chatroom Integration:** Posts detected spam to designated chatrooms for review and action.
*   **Flexible Configuration:** Easily configurable to suit your specific needs.
*   **Multiple Deployment Options:** Supports setup via command line, virtual environments, and Docker containers.
*   **Open Source:**  Licensed under the Apache License, Version 2.0 or the MIT license.

## Getting Started

### Setting Up SmokeDetector

Choose your preferred method to set up SmokeDetector:

*   **Basic Setup:**
    1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
    2.  Navigate to the directory: `cd SmokeDetector`
    3.  Checkout deploy `git checkout deploy`
    4.  Install dependencies: `sudo pip3 install -r requirements.txt --upgrade` and `pip3 install --user -r user_requirements.txt --upgrade`
    5.  Create a `config` file by copying `config.sample`.
    6.  Edit `config` with your desired settings.
    7.  Run the bot: `python3 nocrash.py` (recommended for continuous operation).

*   **Virtual Environment Setup:**
    1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
    2.  Navigate to the directory: `cd SmokeDetector`
    3.  Configure git `git config user.email "smokey@erwaysoftware.com" && git config user.name "SmokeDetector"`
    4.  Checkout deploy `git checkout deploy`
    5.  Create and activate a virtual environment: `python3 -m venv env`
    6.  Install dependencies: `env/bin/pip3 install -r requirements.txt --upgrade` and `env/bin/pip3 install --user -r user_requirements.txt --upgrade`
    7.  Create a `config` file by copying `config.sample`.
    8.  Edit `config` with your desired settings.
    9.  Run the bot: `env/bin/python3 nocrash.py`

*   **Docker Setup:**
    1.  Build a Docker image: (see the original README for details)
    2.  Create a container from the image: (see the original README for details)
    3.  Copy and configure `config`.
    4.  (Optional) Customize the container.
    5.  Start the container.
    6.  (Optional) Use Docker Compose.

**Detailed setup instructions and troubleshooting are available in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).**

## Requirements

*   Stack Exchange account for login.
*   Python versions as defined by Python's [supported phase of the Python life cycle](https://devguide.python.org/versions/).
*   Git 1.8+ (recommended 2.11+) for blacklist/watchlist modifications.

## Blacklist Removal

If you are an official representative of a website seeking removal from the blacklist, please consult the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" documentation.

## License

SmokeDetector is licensed under the Apache License, Version 2.0, or the MIT license, at your option. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) files for details.