# MoviePilot: Automate Your Movie Workflow with Ease

MoviePilot simplifies and streamlines your movie management tasks, offering a user-friendly experience built for automation and efficiency, and it's all available at the [original repository](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

*Disclaimer: This project is intended for learning and personal use only.  Please refrain from promoting this project on any platforms within China.*

## Key Features

*   **Modern Architecture:** Built with a front-end and back-end separation, using FastAPI and Vue3 for a clean and responsive user experience. Frontend: [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend).
*   **Focus on Core Needs:**  Simplified features and settings, with sensible defaults to reduce complexity and improve ease of use.
*   **Intuitive User Interface:** A redesigned user interface that is more beautiful and easier to navigate.
*   **Docker Support**: Ready to be deployed using Docker

## Getting Started

*   **Official Wiki:**  Comprehensive documentation is available on the [MoviePilot Wiki](https://wiki.movie-pilot.org).
*   **API Documentation:** Explore the API at [https://api.movie-pilot.org](https://api.movie-pilot.org).

## Installation and Setup

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy platform-specific libraries (`.so`, `.pyd`, `.bin`) from the `MoviePilot-Resources/resources` directory to the `MoviePilot/app/helper` directory.

3.  **Backend Setup:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend server will run on port 3001 by default.
    *   Access the API documentation at `http://localhost:3001/docs`.

4.  **Frontend Setup:**
    ```bash
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    *   Access the frontend at `http://localhost:5173`.

## Plugin Development

*   Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) in the Wiki to create custom plugins within the `app/plugins` directory.

## Contributing

Contributions are welcome!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>