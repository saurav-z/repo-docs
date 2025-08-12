# MoviePilot: Your Automated Movie & Media Management Solution

MoviePilot is a powerful and user-friendly application designed to streamline your movie and media library management.  Explore the original project on GitHub: [MoviePilot](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

## Key Features

*   **Frontend & Backend Separation:** Built with a modern architecture using FastAPI (backend) and Vue3 (frontend) for enhanced performance and flexibility.
*   **Simplified Design:**  Focuses on core automation needs, reducing complexity and offering sensible defaults for ease of use.
*   **Intuitive User Interface:** Experience a redesigned, visually appealing, and user-friendly interface.
*   **Extensible with Plugins:** Customize and extend functionality through plugin development.

## Installation and Usage

For detailed instructions on installation and configuration, please refer to the official Wiki: [MoviePilot Wiki](https://wiki.movie-pilot.org)

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Installation Steps

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the necessary `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory (matching your platform and version) to the `MoviePilot/app/helper` directory.
3.  **Install Backend Dependencies & Run Backend:**
    ```shell
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend will start on port 3001 by default.  Access the API documentation at `http://localhost:3001/docs`.
4.  **Clone and Run the Frontend Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    The frontend will be accessible at `http://localhost:5173`.
5.  **Develop Plugins:** Consult the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to develop plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is for learning and educational purposes only.
*   It is not intended for commercial use or any illegal activities.
*   The developers are not responsible for user actions.
*   The code is open-source.  Modifications that circumvent restrictions and lead to misuse are the responsibility of the modifier.
*   Donations are not accepted.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>