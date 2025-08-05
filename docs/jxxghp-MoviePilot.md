# MoviePilot: Your Automated Media Management Solution

MoviePilot empowers you to effortlessly manage your media library with an intuitive and streamlined interface.  Check out the [original repository](https://github.com/jxxghp/MoviePilot) for more details and to get started.

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/network/members)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/graphs/contributors)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

**Disclaimer:** This project is intended for learning and personal use only. Please do not promote this project on any domestic (China) platforms.

For updates and announcements, join our channel: [Telegram Channel](https://t.me/moviepilot_channel)

## Key Features

*   **Frontend & Backend Separation:** Built with FastAPI (backend) and Vue3 (frontend) for a modern and responsive user experience.  Frontend: [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend). API Documentation: [http://localhost:3001/docs](http://localhost:3001/docs)
*   **Simplified Core Functionality:**  MoviePilot focuses on essential automation tasks, reducing complexity and improving maintainability.  Many settings utilize sensible defaults for ease of use.
*   **User-Friendly Interface:** Enjoy a redesigned user interface that is both aesthetically pleasing and intuitive to navigate.
*   **Docker Support:** Easily deploy and manage MoviePilot with Docker.

## Installation and Usage

Comprehensive instructions and guides are available on the official Wiki: [MoviePilot Wiki](https://wiki.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Steps to Contribute

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone Resource Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the necessary platform-specific library files (`.so`, `.pyd`, `.bin`) from the `MoviePilot-Resources/resources` directory into the `MoviePilot/app/helper` directory.

3.  **Install Backend Dependencies and Run:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend service will run on port `3001` by default. Access the API documentation at `http://localhost:3001/docs`.

4.  **Clone and Run Frontend:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.

5.  **Plugin Development:** Follow the instructions in the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to develop plugins in the `app/plugins` directory.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>