# MoviePilot: Automate Your Movie and Media Management

MoviePilot is a streamlined media management solution designed for automation and ease of use.  Check out the original project on GitHub: [jxxghp/MoviePilot](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/network/members)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/graphs/contributors)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

## Key Features

*   **Simplified and Focused:** Concentrates on core automation needs, simplifying functionality and settings for a more user-friendly experience.
*   **Modern Architecture:** Built with a front-end and back-end separation using FastAPI and Vue3.
*   **Enhanced User Interface:** Features a redesigned, more aesthetically pleasing, and intuitive user interface.
*   **Easy to Extend:**  Designed for maintainability and extensibility, with a focus on plugin development.

## Installation and Usage

For detailed installation and usage instructions, please refer to the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Contributing

MoviePilot is an open-source project and welcomes contributions.

### Development Requirements

*   Python 3.12
*   Node.js v20.12.1

### Getting Started

1.  Clone the main project:
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  Clone the resources project:
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the necessary `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory to the `MoviePilot/app/helper` directory based on your platform.
3.  Install backend dependencies:
    ```shell
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend service will start on port 3001.  API documentation is available at `http://localhost:3001/docs`.
4.  Clone the frontend project:
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```
5.  Install frontend dependencies and run the frontend:
    ```shell
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.
6.  Develop plugins in the `app/plugins` directory, following the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev).

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>

## Disclaimer

This project is intended for learning and discussion purposes only. Please refrain from promoting this project on any domestic platforms.