# MoviePilot: Your Automated Media Management Solution

[Get the code on GitHub!](https://github.com/jxxghp/MoviePilot)

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/network/members)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/graphs/contributors)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined and user-friendly application built for automated media management. Designed for simplicity and extensibility, MoviePilot provides a powerful solution for managing your media library.

**Disclaimer:** This project is intended for learning and discussion purposes only.  Please refrain from promoting this project on any platforms in China.

## Key Features

*   **Frontend & Backend Separation:** Built with FastAPI (backend) and Vue3 (frontend), offering a clear separation of concerns and improved maintainability.
*   **Focus on Core Functionality:**  MoviePilot simplifies features and settings, prioritizing essential automation tasks and offering sensible defaults.
*   **Enhanced User Interface:** Experience a redesigned, more intuitive, and visually appealing user interface.
*   **Easy to Extend:** Plugin architecture allows for flexible customization and expansion of capabilities.

## Installation and Usage

Detailed instructions for installing and using MoviePilot can be found in the official Wiki:  [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node JS v20.12.1

### Setup

1.  **Clone the main repository:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the resources repository and copy necessary files:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the platform-specific `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory to the `app/helper` directory in the main project.

3.  **Install backend dependencies and run the server:**
    ```bash
    cd MoviePilot/
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend will run on port `3001`. Access the API documentation at `http://localhost:3001/docs`.

4.  **Clone the frontend repository and run the application:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend/
    yarn
    yarn dev
    ```
    The frontend will be accessible at `http://localhost:5173`.

5.  **Develop Plugins:**  Consult the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create your own plugins within the `app/plugins` directory.

## Contributing

We welcome contributions!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>