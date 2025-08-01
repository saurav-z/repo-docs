# MoviePilot: The Ultimate Automated Movie Management Solution

[MoviePilot](https://github.com/jxxghp/MoviePilot) is a powerful and streamlined movie management solution designed for automation, ease of use, and expandability. This project is built upon the foundation of [NAStool](https://github.com/NAStool/nas-tools) with a focus on core automation needs, minimizing complexity, and enhancing maintainability.

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

**Disclaimer:** This project is intended for educational and personal use only. Please refrain from promoting this project on any domestic platforms.

Stay updated with the latest news and announcements through our Telegram channel: [https://t.me/moviepilot_channel](https://t.me/moviepilot_channel)

## Key Features of MoviePilot

*   **Modern Architecture:** Built with a front-end (Vue3) and back-end (FastAPI) separation for enhanced flexibility and scalability.  Access the API documentation at [http://localhost:3001/docs](http://localhost:3001/docs).
*   **Core Automation Focus:** Streamlined functionality and simplified configurations, with sensible defaults to get you up and running quickly.
*   **Improved User Interface:**  A redesigned, user-friendly interface for a better user experience.
*   **Docker Support**: Easily deploy using Docker, including pre-built images.

## Installation and Usage

Refer to the official Wiki for detailed installation and usage instructions: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development Setup

To contribute to MoviePilot, you'll need:

*   `Python 3.12`
*   `Node JS v20.12.1`

Follow these steps to set up your development environment:

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the required platform-specific library files (.so, .pyd, .bin) from the `MoviePilot-Resources/resources` directory to the `MoviePilot/app/helper` directory.

3.  **Install Backend Dependencies and Run Backend:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend service will start on port `3001`. API documentation is available at `http://localhost:3001/docs`.

4.  **Clone and Run Frontend:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.

5.  **Develop Plugins:** Follow the instructions in the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create custom plugins in the `app/plugins` directory.

## Contributors

A big thank you to all our contributors!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>