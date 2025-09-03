# MoviePilot: Your All-in-One Automation Hub for Media Management

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined and user-friendly solution for automating your media management workflows, built with a focus on simplicity and extensibility. This project is based on [NAStool](https://github.com/NAStool/nas-tools) but has been redesigned to provide a more focused and maintainable automation experience.

**Important Note:**  This project is intended for educational and personal use only. Please refrain from promoting or distributing this project on any domestic Chinese platforms.

*   **[View the original repository on GitHub](https://github.com/jxxghp/MoviePilot)**

## Key Features

*   **Frontend/Backend Separation:** Built with FastAPI (backend) and Vue3 (frontend) for a clear separation of concerns.
*   **Simplified Configuration:** Designed for ease of use with streamlined settings and sensible defaults.
*   **Modern User Interface:** Features a redesigned and intuitive user interface for an improved user experience.
*   **Extensible via Plugins:** Easily expand functionality with a plugin system.

## Installation and Usage

Detailed installation and usage instructions are available on the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Development Setup

1.  **Clone the Main Repository:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Repository:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the platform-specific library files (`.so`, `.pyd`, `.bin`) from the `MoviePilot-Resources/resources` directory to the `app/helper` directory within the main project.
3.  **Install Backend Dependencies:**
    ```shell
    cd MoviePilot/
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend service will run on `http://localhost:3001` (API Documentation: `http://localhost:3001/docs`).
4.  **Clone the Frontend Repository:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```
5.  **Install Frontend Dependencies and Run:**
    ```shell
    cd MoviePilot-Frontend/
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.
6.  **Plugin Development:**  Refer to the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is intended for learning and personal use only.
*   It is not intended for commercial use or illegal activities. Users are solely responsible for their actions.
*   The project is open-source. Modifications that circumvent limitations or restrictions are discouraged.
*   Contributions are welcome.
*   This project does not accept donations and does not provide paid services.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>