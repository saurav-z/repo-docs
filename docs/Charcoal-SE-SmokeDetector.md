# SmokeDetector: Real-time Spam Detection for Stack Exchange

**SmokeDetector is a headless chatbot that helps identify and report spam on Stack Exchange in real-time, keeping communities clean.** Find out more and contribute on the [original GitHub repository](https://github.com/Charcoal-SE/SmokeDetector).

## Key Features

*   **Real-time Spam Detection:** Monitors the Stack Exchange realtime tab for new questions and answers.
*   **Automated Chat Reporting:** Posts detected spam to chatrooms for community review.
*   **Uses ChatExchange and Stack Exchange API:** Leverages established APIs for efficient data retrieval and interaction.
*   **Flexible Deployment:** Supports setup via basic installation, virtual environments, and Docker containers.
*   **Docker Compose Support:** Streamlines deployment with a ready-to-use Docker Compose configuration.

## Setup and Usage

SmokeDetector offers several setup options to accommodate your needs. Detailed documentation can be found in the [wiki](https://charcoal-se.org/smokey).

### Basic Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Checkout the deploy branch: `git checkout deploy`
4.  Install dependencies (using pip3):
    ```shell
    sudo pip3 install -r requirements.txt --upgrade
    pip3 install --user -r user_requirements.txt --upgrade
    ```
5.  Copy `config.sample` to `config` and configure your settings.
6.  Run SmokeDetector: `python3 nocrash.py` (recommended) or `python3 ws.py`.

### Virtual Environment Setup

1.  Clone the repository: `git clone https://github.com/Charcoal-SE/SmokeDetector.git`
2.  Navigate to the directory: `cd SmokeDetector`
3.  Configure Git: `git config user.email "smokey@erwaysoftware.com"; git config user.name "SmokeDetector"`
4.  Checkout the deploy branch: `git checkout deploy`
5.  Create a virtual environment: `python3 -m venv env`
6.  Install dependencies:
    ```shell
    env/bin/pip3 install -r requirements.txt --upgrade
    env/bin/pip3 install --user -r user_requirements.txt --upgrade
    ```
7.  Copy `config.sample` to `config` and configure your settings.
8.  Run SmokeDetector: `env/bin/python3 nocrash.py`

### Docker Setup

1.  Build the Docker image:
    ```shell
    DATE=$(date +%F)
    mkdir temp
    cd temp
    wget https://raw.githubusercontent.com/Charcoal-SE/SmokeDetector/master/Dockerfile
    docker build -t smokey:$DATE .
    ```
2.  Create a Docker container: `docker create --name=mysmokedetector smokey:$DATE`
3.  Start the container (after configuration):
    *   Copy `config.sample` to `config` and configure your settings.
    *   Copy the config file into the container: `docker cp config mysmokedetector:/home/smokey/SmokeDetector/config`
4.  (Optional) Access the container's shell for further setup: `docker exec -it mysmokedetector bash` and then `touch ~smokey/ready`
5.  To run smokey, the `ready` file must be created under `/home/smokey`.

#### Automate Docker deployment with Docker Compose

1.  Create a `config` file using [the sample](config.sample).
2.  Create a directory and place the `config` file and `docker-compose.yml` file.
3.  Run `docker-compose up -d` to start SmokeDetector.
4.  Customize with `docker-compose.yml` (Optional) Add constraints for resources such as CPU and memory.

## Requirements

*   Supports Stack Exchange logins.
*   Supports Python versions between "First release" and "End of life" as defined by the [Python life cycle](https://devguide.python.org/versions/).
*   Requires Git 1.8 or higher, with Git 2.11+ recommended.

## Blacklist Removal

If you are an official representative of a website listed on the blacklist and would like to request its removal, please see the [Process for blacklist removal](https://charcoal-se.org/smokey/Process-for-blacklist-removal).

## License

SmokeDetector is available under the following licenses, at your option:

*   Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
*   MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

### Contribution Licensing

By submitting your contribution for inclusion, you agree that it be dual licensed as above, without any additional terms or conditions.