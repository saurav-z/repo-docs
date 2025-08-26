# MoviePilot: Your Automated Movie & Media Management Solution

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a cutting-edge, open-source project designed to automate and simplify your movie and media management needs, offering a streamlined and user-friendly experience.  Based on selected code from [NAStool](https://github.com/NAStool/nas-tools), MoviePilot focuses on core automation features, aiming for a stable, extensible, and easy-to-use solution.

**Important Note:** This project is intended for educational and personal use only. Please refrain from promoting this project on any platforms within China.

**Stay Updated:** Join the MoviePilot community on Telegram: [https://t.me/moviepilot_channel](https://t.me/moviepilot_channel)

## Key Features

*   **Modern Architecture:** Built with a front-end and back-end separation using FastAPI (backend) and Vue3 (frontend) for enhanced performance and maintainability.
*   **Focus on Simplicity:** Streamlined functionality and settings, designed for ease of use.  Many settings have sensible defaults.
*   **Intuitive User Interface:** A redesigned user interface provides a more beautiful and user-friendly experience.
*   **Docker Support:** Available as Docker images for easy deployment and management.

## Getting Started

### Installation

MoviePilot requires Python 3.12 and Node.js v20.12.1.

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the platform-specific library files (`.so`, `.pyd`, `.bin`) from the `MoviePilot-Resources/resources` directory to the `app/helper` directory within the main project.
3.  **Install Backend Dependencies:**
    ```bash
    cd MoviePilot  # Navigate to the main project directory
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend will run on port 3001 by default.  API Documentation can be accessed at: `http://localhost:3001/docs`
4.  **Clone the Frontend Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```
5.  **Install Frontend Dependencies and Run:**
    ```bash
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    *   Access the frontend at: `http://localhost:5173`
6.  **Plugin Development:**  Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create custom plugins within the `app/plugins` directory.

### Contributing

See the official [API documentation](https://api.movie-pilot.org) to learn how to contribute.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend) - The front-end user interface.
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources) - Contains resources required by MoviePilot.
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins) - Plugin development.
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server) - Backend server.
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki) - Official documentation and wiki.

## Disclaimer

*   This software is intended for educational and personal use only.
*   It must not be used for commercial purposes or illegal activities.
*   The developers are not responsible for the user's actions.
*   The source code is open-source.
*   The project does not accept donations and does not offer paid services.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>