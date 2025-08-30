# MoviePilot: Your All-in-One Automation Solution for Media Management

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a powerful and user-friendly application designed to streamline and automate your media management needs. This project, inspired by NAStool, focuses on core automation tasks, offering a more streamlined and extensible solution.  **[Check out the original project on GitHub](https://github.com/jxxghp/MoviePilot)!**

## Key Features

*   **Modern Architecture:** Built with a front-end and back-end separation using FastApi and Vue3, enabling better maintainability and scalability.
*   **Simplified Configuration:** Focuses on essential features, with many settings defaulting to optimal values for ease of use.
*   **Enhanced User Interface:** Features a redesigned, intuitive, and visually appealing user interface for a superior user experience.
*   **Docker Support:**  Ready to use with Docker.
*   **Plugin Architecture:**  Extensible via plugin support.

## Installation and Usage

Comprehensive guides are available on the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node JS v20.12.1

### Setup Instructions

1.  **Clone the Main Repository:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Repository:**  Fetch platform-specific libraries.
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the necessary `.so`/`.pyd`/`.bin` files from the `resources` directory (corresponding to your platform and version) to the `app/helper` directory within the main project.
3.  **Install Backend Dependencies:**
    ```shell
    cd app
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend service will start on port `3001` by default. Access the API documentation at `http://localhost:3001/docs`.
4.  **Clone the Frontend Repository:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```
5.  **Install Frontend Dependencies and Run:**
    ```shell
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.
6.  **Plugin Development:**  Follow the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev) to create your plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is intended for learning and personal use only. Commercial use and distribution are strictly prohibited.
*   The developers are not responsible for any misuse of the software or any illegal activities performed using this software.
*   The code is open-source. Users are responsible for any modifications to the code and any consequences arising from such modifications. It is not recommended to circumvent or alter the user authentication mechanism.
*   The project does not accept donations and does not provide any paid services. Please be cautious and avoid any misleading offers.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>