# MoviePilot: Automate Your Media Workflow with Ease

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined, open-source media automation tool designed to simplify and enhance your media management experience.  This project is built upon the foundation of [NAStool](https://github.com/NAStool/nas-tools), with a focus on core automation needs.

**[Click here to view the original repository on GitHub](https://github.com/jxxghp/MoviePilot).**

## Key Features

*   **Modern Architecture:**  Built with a front-end (Vue3) and back-end (FastAPI) separation for enhanced performance and maintainability.
*   **Focused Automation:** Designed to streamline core automation tasks with simplified settings.
*   **User-Friendly Interface:**  Features a redesigned, intuitive, and visually appealing user interface.
*   **Extensible with Plugins:** Designed to allow the addition of custom functionality through plugins.

## Installation and Usage

Comprehensive installation and usage instructions can be found in the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Steps

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone Resource Project and Copy Libraries:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the relevant `.so`, `.pyd`, or `.bin` files from `MoviePilot-Resources/resources` (matching your platform and version) to the `app/helper` directory within the main project.
3.  **Install Backend Dependencies and Run the Backend:**
    ```shell
    cd app  # Navigate to the app directory if needed
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend server will run on port 3001 by default. Access the API documentation at `http://localhost:3001/docs`.
4.  **Clone and Run the Frontend:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    The frontend will be available at `http://localhost:5173`.
5.  **Plugin Development:** Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) for instructions on creating plugins within the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is intended for educational and personal use only.
*   It is strictly prohibited to use this software for commercial purposes or any illegal activities. Users are solely responsible for their actions.
*   The software is open-source. Any modifications or distributions of the code are the responsibility of the modifier/distributor.  Circumventing or modifying user authentication mechanisms for public distribution is strongly discouraged.
*   This project does not accept donations and does not offer any paid services.  Please exercise caution to avoid any potential scams.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>