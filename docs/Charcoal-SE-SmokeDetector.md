# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Tired of spam cluttering your Stack Exchange chatrooms?** SmokeDetector is a powerful, headless chatbot that automatically identifies and reports spam, keeping your communities clean and efficient.  Visit the [original repository](https://github.com/Charcoal-SE/SmokeDetector) for more details.

[![Build Status](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml/badge.svg?query=branch%3Amaster)](https://github.com/Charcoal-SE/SmokeDetector/actions/workflows/build.yml?query=branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/Charcoal-SE/SmokeDetector/badge.svg?branch=master)](https://coveralls.io/github/Charcoal-SE/SmokeDetector?branch=master)
[![Open issues](https://img.shields.io/github/issues/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Charcoal-SE/SmokeDetector.svg)](https://github.com/Charcoal-SE/SmokeDetector/pulls)

## Key Features

*   **Automated Spam Detection:**  SmokeDetector analyzes Stack Exchange questions in real-time.
*   **Chat Integration:** Posts detected spam directly to chatrooms for immediate visibility and action.
*   **Uses Official APIs:** Leverages the Stack Exchange API and ChatExchange for seamless data retrieval and communication.
*   **Highly Configurable:**  Easily customize SmokeDetector to fit your specific community needs through its config file.
*   **Multiple Deployment Options:**  Supports setup using standard Python, virtual environments, and Docker for flexible deployment.

## Getting Started

### Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    ```

2.  **Checkout the deploy branch**
    ```bash
    git checkout deploy
    ```

3.  **Install Dependencies:**
    ```bash
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```

4.  **Configure:**
    *   Copy `config.sample` to a new file named `config`.
    *   Edit the values in `config` according to your needs.

5.  **Run:**
    *   Use `python3 nocrash.py` (recommended for continuous operation) or `python3 ws.py`.

### Deployment Options

*   **Standard Python:** Follow the basic setup instructions above.

*   **Virtual Environment:**  Isolate dependencies for cleaner setups.
    1.  Follow the basic steps above.
    2.  Create a virtual environment: `python3 -m venv env`
    3.  Activate your environment and install dependencies.
    4.  Run SmokeDetector with `env/bin/python3 nocrash.py`.

*   **Docker:**  The most robust way to isolate your dependencies.
    1.  Follow the basic setup steps above.
    2.  Build the Docker image (refer to the original README).
    3.  Create and start a container from the image.
    4.  Copy your `config` file into the container.
    5.  Automated Docker deployment via Docker Compose is also available.

## Requirements

*   **Python:** Supports versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/).
*   **Stack Exchange Login:** Requires a Stack Exchange login.
*   **Git (for Blacklist/Watchlist Modifications):** Git 1.8+ (2.11+ recommended).

## Documentation

Detailed user documentation is available in the [wiki](https://charcoal-se.org/smokey), including [setting up and running SmokeDetector](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

## Blacklist Removal

If you are an official representative and want to request website removal, see the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)".

## License

Licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) or the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>) at your option.

### Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.