# MoviePilot: Your Ultimate Movie Automation Companion

MoviePilot is a powerful, open-source movie automation tool designed for simplifying your media management workflow.  [Learn more on GitHub](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)


## Key Features

*   **Modern Architecture:** Built with a frontend/backend separation using FastAPI and Vue3 for enhanced performance and maintainability.
*   **Simplified Configuration:** Focused on core automation needs, minimizing complex settings and offering sensible defaults for ease of use.
*   **User-Friendly Interface:** A redesigned, intuitive, and visually appealing user interface for a seamless experience.
*   **Extensible with Plugins**: Easily expand functionality with custom plugins (plugin development guide available).
*   **Docker Support**: Available as Docker images for simplified deployment.

## Installation and Usage

For detailed installation instructions and usage guides, please refer to the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development Setup

To contribute to MoviePilot or develop custom plugins, follow these steps:

**Prerequisites:**  `Python 3.12`, `Node JS v20.12.1`

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the `.so`, `.pyd`, or `.bin` files from the `resources` directory (matching your platform and version) into the `app/helper` directory.
3.  **Install Backend Dependencies:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend service will run on port `3001` by default.
    *   API documentation is available at: `http://localhost:3001/docs`
4.  **Clone and Run the Frontend:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    *   Access the frontend at: `http://localhost:5173`
5.  **Plugin Development:**
    *   Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is intended for learning and personal use only.  It should not be used for commercial purposes or illegal activities. The software developers are not responsible for user actions.
*   The source code is open-source. Modifying and redistributing the code without proper authorization may result in legal issues. Do not circumvent or modify the user authentication mechanisms for public release.
*   This project does not accept donations, nor does it offer paid services. Please be cautious of any misleading information regarding donations or paid features.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>