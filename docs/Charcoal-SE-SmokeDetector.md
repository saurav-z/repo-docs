# SmokeDetector: Real-time Spam Detection for Stack Exchange

**Tired of spam cluttering your Stack Exchange chatrooms?** SmokeDetector is a headless chatbot that swiftly identifies and reports spam, keeping your communities clean.  Learn more about this project on [GitHub](https://github.com/Charcoal-SE/SmokeDetector).

---

## Key Features:

*   **Real-time Spam Detection:** Monitors Stack Exchange's real-time feed for spam.
*   **Chatroom Reporting:** Posts detected spam to designated chatrooms for moderation.
*   **Uses ChatExchange:**  Leverages the ChatExchange library for seamless integration with Stack Exchange chat.
*   **Stack Exchange API Integration:** Accesses question and answer data via the Stack Exchange API.
*   **Docker Support**: Easily deploy and manage with Docker for isolated environments.
*   **Virtual Environment Support**:  Set up SmokeDetector using Python virtual environments for streamlined dependency management.

---

## Getting Started

### Basic Setup

1.  **Clone the Repository:** `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  **Navigate to Directory:** `cd SmokeDetector`
3.  **Configure:** Copy `config.sample` to `config` and edit with your settings.
4.  **Install Dependencies:**
    ```bash
    git checkout deploy
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
5.  **Run the Bot:** `python3 nocrash.py` (recommended for continuous operation) or `python3 ws.py` (shuts down after 6 hours).

### Virtual Environment Setup

1.  **Clone and Configure Git:**
    ```bash
    git clone https://github.com/Charcoal-SE/SmokeDetector.git
    cd SmokeDetector
    git config user.email "smokey@erwaysoftware.com"
    git config user.name "SmokeDetector"
    git checkout deploy
    ```
2.  **Create and Activate Environment:**
    ```bash
    python3 -m venv env
    ```
3.  **Install Dependencies:**
    ```bash
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
4.  **Configure:** Copy `config.sample` to `config` and edit.
5.  **Run the Bot:** `env/bin/python3 nocrash.py`

### Docker Setup

1.  **Build Docker Image:**
    ```bash
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  **Create Docker Container:** `docker create --name=mysmokedetector smokey:$DATE`
3.  **Configure:** Copy `config.sample` to `config` and edit within the container: `docker cp config mysmokedetector:/home/smokey/SmokeDetector/config`
4.  **Start the Container:** Once `config` is ready, create the file `/home/smokey/ready` by running: `touch ~smokey/ready`.
5.  **Run Commands in Container**: `docker exec -it mysmokedetector bash`

#### Automate Docker deployment with Docker Compose

1.  **Create `config` file:** Follow the basic configuration steps.
2.  **Create `docker-compose.yml` file:** Place the `config` file and the `docker-compose.yml` file (from the original repo) in a directory.
3.  **Start the Service:**  `docker-compose up -d`

---

## Requirements

*   Stack Exchange Login (used for user authentication)
*   Python versions:  Support for Python versions within the [supported phase of the Python life cycle](https://devguide.python.org/versions/) (as defined as between "First release" and "End of life").
*   Git 1.8 or higher (Git 2.11+ recommended) for blacklist/watchlist modifications.

---

## Blacklist Removal Process

For official website/product representatives seeking blacklist removal, please consult the "[Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal)" for detailed instructions.

---

## License

This project is available under the following licenses:

*   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

See the LICENSE files for full details.

### Contribution Licensing

By submitting your contribution, you agree that it be dual licensed as above,
without any additional terms or conditions.