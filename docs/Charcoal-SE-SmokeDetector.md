# SmokeDetector: Real-Time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot that identifies and reports spam in real-time on the Stack Exchange network.**

[View the original repository on GitHub](https://github.com/Charcoal-SE/SmokeDetector)

SmokeDetector uses advanced techniques to automatically identify and report spam, keeping Stack Exchange communities clean and safe. Here's what it offers:

*   **Real-time Spam Detection:** Monitors the Stack Exchange platform for suspicious activity.
*   **Automated Reporting:** Posts detected spam to designated chatrooms for review.
*   **Open Source:** Contribute to the project and help improve its spam-fighting capabilities.
*   **Flexible Setup:** Supports multiple deployment methods including Docker, virtual environments, and direct installation.

## Getting Started

Detailed setup and running instructions can be found in the [wiki](https://charcoal-se.org/smokey).

### Installation Options

Choose your preferred setup method:

*   **Basic Setup:**
    1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
    2.  Navigate to the directory: `cd SmokeDetector`
    3.  Checkout deploy branch: `git checkout deploy`
    4.  Install dependencies:
        ```shell
        sudo pip3 install -r requirements.txt --upgrade
        pip3 install --user -r user_requirements.txt --upgrade
        ```
    5.  Copy and configure the configuration file:  Copy `config.sample` to a new file called `config`, and edit the values required.
    6.  Run SmokeDetector: `python3 nocrash.py` (recommended for continuous operation) or `python3 ws.py` (shuts down after 6 hours).

*   **Virtual Environment Setup:** This is recommended for isolating dependencies.  Follow the instructions in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

*   **Docker Setup:** The recommended way to isolate dependencies. Instructions on how to use a [Dockerfile](Dockerfile), and use Docker Compose can also be found in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

## Requirements

*   **Python:** Compatible with Python versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/).
*   **Git:** Git 1.8 or higher (Git 2.11+ recommended) for blacklist/watchlist modifications.
*   **Stack Exchange Login:** Requires a Stack Exchange login.

## Blacklist Removal Process

For website/product representatives seeking removal from the blacklist, please see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" in the wiki for details.

## License

SmokeDetector is licensed under the following:

*   Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

Contributions are dual-licensed under the same terms.