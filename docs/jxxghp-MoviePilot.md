# MoviePilot: Your Automation Hub for Enhanced Media Management

MoviePilot is a powerful and user-friendly application designed to streamline and automate your media management tasks. Explore the original repository on [GitHub](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

## Key Features

*   **Frontend and Backend Separation:**  Built with a clear separation between the user interface (Vue3) and the backend (FastAPI), for enhanced maintainability and scalability.
*   **Focused Core Functionality:** Designed to streamline essential automation tasks, simplifying settings and offering sensible defaults.
*   **Intuitive User Interface:**  A redesigned and user-friendly interface for an improved user experience.

## Installation and Usage

Comprehensive guides and documentation are available on the [official Wiki](https://wiki.movie-pilot.org).

## Prerequisites

*   Python 3.12
*   Node.js v20.12.1

## Getting Started

1.  **Clone the Main Project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project:**  Get platform-specific library files from [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources) and copy them into `app/helper`.

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```

3.  **Install Backend Dependencies and Run:**

    ```bash
    pip install -r requirements.txt
    python3 main.py
    ```

    *   The backend service will run on port `3001` by default.
    *   Access the API documentation at: `http://localhost:3001/docs`

4.  **Clone and Run the Frontend:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```

    *   Access the frontend at `http://localhost:5173`

5.  **Plugin Development:** Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is for educational and personal use only.
*   Please use responsibly and be aware of any local regulations.
*   The developers are not responsible for user actions.
*   Contribution guidelines exist to avoid circumventing any limitations.
*   This project does not accept donations.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>