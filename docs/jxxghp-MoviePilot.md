# MoviePilot: Your Automated Movie and Media Management Solution

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined, open-source media management system designed for automation and ease of use, built with a focus on core functionality. Based on parts of [NAStool](https://github.com/NAStool/nas-tools), MoviePilot simplifies your media workflow and is designed for easy expansion and maintenance.

## Key Features

*   **Frontend and Backend Separation:** Built with FastAPI for the backend and Vue3 for the frontend ([MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)), providing a responsive and modern user experience.  API documentation is available at: `http://localhost:3001/docs`.
*   **Core Functionality Focused:**  Prioritizes essential features, simplifying settings and offering sensible defaults for easy setup.
*   **User-Friendly Interface:**  A redesigned user interface for a more beautiful and intuitive experience.

## Installation and Usage

For detailed installation instructions and usage guides, please visit the official MoviePilot Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development and Contribution

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Development Setup

1.  **Clone the main project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the resources project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the necessary `.so`/`.pyd`/`.bin` files from the `MoviePilot-Resources/resources` directory to the `app/helper` directory in the main project, matching your platform and version.

3.  **Install Backend Dependencies and Run:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend service will start on port `3001`. Access the API documentation at: `http://localhost:3001/docs`.

4.  **Clone and Run Frontend:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.

5.  **Plugin Development:**  Refer to the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins within the `app/plugins` directory.

## Contributors

[<img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />](https://github.com/jxxghp/MoviePilot/graphs/contributors)

**Disclaimer:** This project is for learning and communication purposes only. Please refrain from promoting this project on any domestic platforms within China.

**Stay Updated:** Join the MoviePilot channel: [https://t.me/moviepilot_channel](https://t.me/moviepilot_channel)

**[Back to the Original Repository](https://github.com/jxxghp/MoviePilot)**