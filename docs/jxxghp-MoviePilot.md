# MoviePilot: Your Open-Source Movie Automation Solution

**MoviePilot** is a streamlined and user-friendly movie automation solution designed to simplify your media management and enhance your viewing experience. ([View the original repository on GitHub](https://github.com/jxxghp/MoviePilot))

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

Based on the original [NAStool](https://github.com/NAStool/nas-tools) but redesigned with a focus on core automation needs, MoviePilot aims to reduce complexity and improve maintainability.

**Important Disclaimer:** This project is intended for educational and personal use only.  Please do not promote this project on any platform in China.

Stay updated: [MoviePilot Telegram Channel](https://t.me/moviepilot_channel)

## Key Features

*   **Frontend and Backend Separation:** Built with FastApi for the backend and Vue3 for the frontend.
*   **Simplified Functionality:** Focuses on core automation tasks, minimizing complex configurations.
*   **Enhanced User Interface:** Features a redesigned, modern, and easy-to-use interface.
*   **Easy to Extend:** Build new functionality with plugins.

## Installation and Usage

Find detailed installation and usage instructions in the official wiki: [MoviePilot Wiki](https://wiki.movie-pilot.org)

### Prerequisites

*   `Python 3.12`
*   `Node JS v20.12.1`

### Installation Steps

1.  **Clone the main project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the resource project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the platform-specific library files (`.so`, `.pyd`, or `.bin`) from the `resources` directory of `MoviePilot-Resources` to the `app/helper` directory of `MoviePilot`.
3.  **Install backend dependencies and run the backend:**
    ```bash
    cd app
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend will start on port `3001`.  Access the API documentation at: `http://localhost:3001/docs`
4.  **Clone the frontend project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```
5.  **Install frontend dependencies and run the frontend:**
    ```bash
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at: `http://localhost:5173`
6.  **Develop Plugins:**  Follow the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is intended for personal learning and educational purposes only.  It should not be used for commercial activities or any illegal purposes. The user assumes all responsibility for their use of the software.
*   The source code is open-source. Modifications that circumvent restrictions and lead to distribution or illegal activities are the responsibility of the code modifier.
*   This project does not accept donations and does not offer paid services.  Please be cautious of any misleading information.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>