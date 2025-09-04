# MoviePilot: Automate Your Movie Workflow with Ease

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined and user-friendly application, born from the foundation of [NAStool](https://github.com/NAStool/nas-tools), designed to automate your movie workflow and enhance your media management experience.

## Key Features

*   **Modern Architecture:** Built with a clean, front-end and back-end separation using FastAPI and Vue3 for a robust and scalable system.
*   **Simplified Configuration:** Focused on core automation needs, MoviePilot reduces complexity with sensible defaults and simplified settings, making setup straightforward.
*   **Enhanced User Interface:** Features a redesigned, intuitive, and visually appealing user interface for an improved user experience.
*   **Platform Support:** Works on Windows, Linux, and Synology platforms.
*   **Docker Support**: Offers Docker images for easy deployment and management.

## Installation and Usage

*   **Official Wiki:** Comprehensive guides and documentation are available at [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org).
*   **API Documentation:** Explore the API at [https://api.movie-pilot.org](https://api.movie-pilot.org).

### Prerequisites

*   Python 3.12
*   Node JS v20.12.1

### Steps

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone Resource Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy platform-specific library files (`.so`, `.pyd`, `.bin`) from `MoviePilot-Resources/resources` to `MoviePilot/app/helper`.
3.  **Install Backend Dependencies and Run:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend runs on port 3001 by default, and the API documentation is accessible at `http://localhost:3001/docs`.
4.  **Clone Frontend Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```
5.  **Install Frontend Dependencies and Run:**
    ```bash
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.
6.  **Plugin Development:**
    Refer to the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev) to create custom plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Important Notices

*   **For Educational Use Only:** This software is intended solely for learning and educational purposes.
*   **No Commercial Use or Distribution in China:** Do not promote or distribute this project on any platforms within China.
*   **Disclaimer:** The developers are not responsible for any user actions or misuse of the software.  Please review the full disclaimer for more details.

## Contributing

We welcome contributions!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>