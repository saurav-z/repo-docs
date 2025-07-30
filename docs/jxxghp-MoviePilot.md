# MoviePilot: Your Automated Movie and Media Management Companion

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined media management solution, built on a foundation of automation, offering a user-friendly experience.  (For a deeper dive, check out the original repository [here](https://github.com/jxxghp/MoviePilot).)

**Important Disclaimer:** This project is intended for learning and personal use only. Please refrain from promoting this project on any domestic platforms.

## Key Features

*   **Modern Architecture:** Built with a decoupled architecture using FastAPI (backend) and Vue3 (frontend) for enhanced flexibility and maintainability.
*   **Simplified Configuration:** Focused on essential features and optimized settings, making it easy to get started.
*   **Intuitive User Interface:** A redesigned and improved UI for a better user experience.
*   **Automated Media Management:** Designed to automate core needs for media management, providing ease of use and extensibility.
*   **Docker Support:** Available as a Docker image for easy deployment.

## Installation and Usage

For detailed installation and usage instructions, please refer to the official MoviePilot Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development Setup

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Steps:

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone Resources Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the required platform-specific library files (`.so`, `.pyd`, or `.bin`) from the `MoviePilot-Resources/resources` directory into the `app/helper` directory of the main project.

3.  **Backend Setup:**
    *   Navigate to the main project directory and install backend dependencies:
        ```shell
        cd <your-moviepilot-directory>
        pip install -r requirements.txt
        ```
    *   Run the backend service (default port: 3001, API documentation: `http://localhost:3001/docs`):
        ```shell
        python3 main.py
        ```

4.  **Frontend Setup:**
    *   Clone the frontend project:
        ```shell
        git clone https://github.com/jxxghp/MoviePilot-Frontend
        ```
    *   Install frontend dependencies and run the frontend service (accessible at `http://localhost:5173`):
        ```shell
        cd <your-moviepilot-frontend-directory>
        yarn
        yarn dev
        ```

5.  **Plugin Development:**
    *   Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to develop plugins in the `app/plugins` directory.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>