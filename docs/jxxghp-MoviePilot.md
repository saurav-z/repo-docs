# MoviePilot: Your Automated Movie Management Companion

MoviePilot is a streamlined and user-friendly application designed to automate your movie management needs, built upon core functionalities and focusing on ease of use and extensibility. Explore the project on [GitHub](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

**Disclaimer:** This project is for learning and discussion purposes only. Please refrain from promoting it on any domestic platforms.

## Key Features

*   **Frontend and Backend Separation:** Utilizes a modern architecture with FastAPI (backend) and Vue3 (frontend).
    *   Frontend: [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
    *   API Documentation: `http://localhost:3001/docs` (default)
*   **Focused on Core Needs:** Simplifies functionalities and settings, making it easier to use. Many settings have sensible default values.
*   **Improved User Interface:** Features a redesigned, more intuitive, and aesthetically pleasing user interface.

## Installation and Usage

For detailed instructions and guidance on getting started with MoviePilot, please consult the official [Wiki](https://wiki.movie-pilot.org).

## Development Setup

To contribute to MoviePilot, you'll need the following:

*   Python 3.12
*   Node.js v20.12.1

**Steps:**

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the appropriate `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory to the `app/helper` directory within the main project.

3.  **Install Backend Dependencies and Run:**
    ```shell
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend service defaults to port `3001`.
    *   API documentation is available at `http://localhost:3001/docs`.

4.  **Clone and Run the Frontend Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    yarn
    yarn dev
    ```
    *   Access the frontend at `http://localhost:5173`.

5.  **Plugin Development:**
    *   Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create custom plugins within the `app/plugins` directory.

## Contributors

A big thank you to our contributors!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>