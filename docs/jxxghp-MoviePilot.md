# MoviePilot: The Ultimate Automation Solution for Media Management

MoviePilot is a powerful and user-friendly application designed to automate and simplify your media management tasks, built upon the foundation of NAS tools.  ([See the original repository](https://github.com/jxxghp/MoviePilot))

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

## Key Features

*   **Modern Architecture:**  Built with a frontend-backend separation using FastAPI and Vue3 for a streamlined and responsive user experience.
*   **Focus on Core Automation:**  Prioritizes essential features and simplifies settings, reducing complexity and improving usability.
*   **Intuitive User Interface:**  Features a redesigned and aesthetically pleasing user interface for easier navigation and control.
*   **Extensible:** Designed for ease of expansion, providing opportunities for custom plugins and functionalities.

## Installation and Usage

Refer to the official wiki for detailed installation and usage instructions: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Getting Started

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone Resources Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the necessary platform-specific libraries (`.so`, `.pyd`, `.bin`) from the `MoviePilot-Resources/resources` directory to the `app/helper` directory within the main project.

3.  **Install Backend Dependencies:**
    ```shell
    cd <MoviePilot project directory>
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend will run on port `3001` by default.  Access the API documentation at `http://localhost:3001/docs`.

4.  **Clone Frontend Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```

5.  **Install Frontend Dependencies and Run:**
    ```shell
    cd <MoviePilot-Frontend project directory>
    yarn
    yarn dev
    ```
    *   Access the frontend at `http://localhost:5173`.

6.  **Plugin Development:**
    *   Develop custom plugins within the `app/plugins` directory.  Refer to the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev) for details.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is for learning and exchange purposes only.
*   It is not for commercial use.
*   The developers are not responsible for user actions or any illegal activities.
*   This project does not accept donations and does not offer any paid services.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>