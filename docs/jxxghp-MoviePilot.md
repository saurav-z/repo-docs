# MoviePilot: Your Automated Media Management Companion

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined media management solution designed for automation, ease of use, and extensibility.  This project is built on [NAStool](https://github.com/NAStool/nas-tools) code, focusing on core automation needs.

**Important Note:** This project is for learning and discussion purposes only. Please refrain from promoting this project on any domestic platforms.

*   **Telegram Channel:** [https://t.me/moviepilot_channel](https://t.me/moviepilot_channel)

## Key Features

*   **Frontend & Backend Separation:** Built with FastAPI (backend) and Vue3 (frontend).
    *   Frontend Repository: [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
    *   API Documentation: http://localhost:3001/docs
*   **Simplified Functionality:** Focuses on core needs, simplifying settings and offering default values where appropriate.
*   **Enhanced User Interface:** A redesigned, more user-friendly, and aesthetically pleasing interface.

## Installation and Usage

Detailed instructions and documentation are available on the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development Setup

Requires `Python 3.12` and `Node JS v20.12.1`.

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the necessary platform-specific library files (`.so`, `.pyd`, or `.bin`) from `MoviePilot-Resources/resources` to the `app/helper` directory in the main project.
3.  **Install Backend Dependencies and Run:** Set `app` as the source code root directory and run `main.py` to start the backend service, listening on port 3001 by default.
    ```shell
    pip install -r requirements.txt
    python3 main.py
    ```
    *   API Documentation: `http://localhost:3001/docs`
4.  **Clone and Run Frontend:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    yarn
    yarn dev
    ```
    *   Access the frontend at: `http://localhost:5173`
5.  **Plugin Development:** Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>