# SmokeDetector: Your Real-Time Spam Hunter for Stack Exchange

Tired of spam infiltrating your Stack Exchange communities? SmokeDetector is a powerful, headless chatbot designed to swiftly detect and report spam in real-time, keeping your platforms clean and community engaged. ([View the original repository](https://github.com/Charcoal-SE/SmokeDetector))

## Key Features:

*   **Real-Time Spam Detection:** Monitors the Stack Exchange realtime tab and API to identify and flag spam.
*   **Automated Reporting:** Posts detected spam to chatrooms for efficient community moderation.
*   **Flexible Setup:**  Supports various deployment methods including standard, virtual environments, and Docker containers.
*   **Customizable Configuration:** Easy to set up and configure to meet specific needs and preferences.
*   **Community Driven:** Actively maintained and improved with community contributions.

## Getting Started

### Prerequisites:

*   Python 3.7+ (or a supported version)
*   Git 1.8+ (2.11+ recommended)
*   Stack Exchange login

### Installation:

Choose your preferred installation method:

*   **Standard Setup:**
    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git checkout deploy
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
    Then, copy `config.sample` to `config` and edit the required values.  Run with `python3 nocrash.py`.

*   **Virtual Environment:**
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
    Copy `config.sample` to `config` and edit. Run with `env/bin/python3 nocrash.py`.

*   **Docker:**
    Follow the provided [Dockerfile](Dockerfile) instructions to build and run SmokeDetector in a container.  For automated deployment, use the provided `docker-compose.yml` example.  Remember to configure the `config` file.

## Documentation

*   Comprehensive user documentation is available in the [wiki](https://charcoal-se.org/smokey).
*   Detailed setup and running guides are found in the [wiki](https://charcoal-se.org/smokey/Set-Up-and-Run-SmokeDetector).

## License

This project is licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>) or the MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>), at your option.

### Contributing

We welcome contributions! Please review the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0) for details on contributing.