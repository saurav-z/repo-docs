# SmokeDetector: Real-time Spam Detection for Stack Exchange

**SmokeDetector is a powerful, headless chatbot that tirelessly monitors Stack Exchange for spam, automatically flagging and reporting it to chatrooms.**

[View the original repository on GitHub](https://github.com/Charcoal-SE/SmokeDetector)

Key features:

*   **Real-time Spam Detection:** Monitors Stack Exchange's real-time feed for suspicious content.
*   **Automated Reporting:** Posts detected spam to chatrooms for community awareness and action.
*   **API Integration:** Leverages the Stack Exchange API to access and analyze question and answer content.
*   **Flexible Setup:** Supports setup via standard installation, virtual environments, and Docker containers for versatile deployment.

## Getting Started

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git checkout deploy
    ```

2.  **Install Dependencies:**

    ```bash
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```

3.  **Configuration:**

    *   Copy `config.sample` to a new file named `config`.
    *   Edit the values in the `config` file with your desired settings.

4.  **Running SmokeDetector:**

    *   Execute `python3 nocrash.py` (recommended for continuous operation) or `python3 ws.py`.

### Virtual Environment Setup

To isolate dependencies, use a virtual environment:

1.  **Set up virtual environment**

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

2.  **Configuration:**

    *   Copy `config.sample` to a new file named `config`.
    *   Edit the values in the `config` file with your desired settings.

3.  **Running SmokeDetector:**

    *   Execute `env/bin/python3 nocrash.py`.

### Docker Setup

For an even more isolated environment:

1.  **Build the Docker Image:**

    ```bash
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```

2.  **Create a Container:**

    ```bash
    docker create --name=mysmokedetector smokey:$DATE
    ```

3.  **Configuration:**

    *   Copy `config.sample` to a new file named `config`.
    *   Edit the values in the `config` file with your desired settings.
    *   Copy the `config` file into the container:

        ```bash
        docker cp config mysmokedetector:/home/smokey/SmokeDetector/config
        ```

4.  **Start the Container:**

    *   Start SmokeDetector: `docker start mysmokedetector`
    *   The bot will now run and monitor for spam

### Docker Compose

For automated deployments:

1.  **Configuration:**
    *   Create the `config` file.
2.  **Create a `docker-compose.yml` file**
3.  **Run `docker-compose up -d`**

## Requirements

*   Stack Exchange login credentials.
*   Python versions in the [supported phase of the Python life cycle](https://devguide.python.org/versions/).
*   Git 1.8 or higher (Git 2.11+ recommended) for blacklist and watchlist modifications.

## Blacklist Removal

If you are an official representative of the website/product which you desire to see removed, please see "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for details as to how to request removal of your website from the blacklist.

## License

SmokeDetector is licensed under either the Apache License, Version 2.0 or the MIT license. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for more details.

## Contribution Licensing

By submitting your contribution for inclusion in the work
as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0),
you agree that it be dual licensed as above,
without any additional terms or conditions.