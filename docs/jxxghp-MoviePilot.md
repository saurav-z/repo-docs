# MoviePilot: Automate Your Media Management (and Learn!)

**MoviePilot** is a powerful media management tool designed for automation, built with a streamlined architecture for ease of use and extensibility.  Find the original project on GitHub:  [jxxghp/MoviePilot](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

**Disclaimer:** This project is intended for learning and educational purposes.  Please be mindful of copyright and distribution regulations in your region.

## Key Features

*   **Modern Architecture:** Built with a front-end (Vue3) and back-end (FastAPI) separation for improved maintainability and user experience.
*   **Streamlined Functionality:** Focuses on core automation needs, simplifying settings and offering sensible defaults.
*   **Enhanced User Interface:** A redesigned, user-friendly interface for a more intuitive experience.
*   **Docker Support:** Deploy easily with Docker images available.

## Getting Started

*   **Official Wiki:**  [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)
*   **API Documentation:** [https://api.movie-pilot.org](https://api.movie-pilot.org)

## Installation & Development

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Setup

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the required `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory to the `app/helper` directory in your main project, matching your platform and version.
3.  **Backend Setup:**

    *   Navigate to the `app` directory (your source code root).
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *   Run the backend service (listening on port 3001 by default):
        ```bash
        python3 main.py
        ```
    *   Access the API documentation at: `http://localhost:3001/docs`
4.  **Frontend Setup:**
    *   Clone the Frontend Project:
        ```bash
        git clone https://github.com/jxxghp/MoviePilot-Frontend
        ```
    *   Install frontend dependencies:
        ```bash
        yarn
        ```
    *   Run the frontend:
        ```bash
        yarn dev
        ```
    *   Access the frontend at: `http://localhost:5173`

5.  **Plugin Development:**
    *   Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Contributing

Contributions are welcome!  Please review the contribution guidelines (if available).

## Contributors

[![Contributors](https://contrib.rocks/image?repo=jxxghp/MoviePilot)](https://github.com/jxxghp/MoviePilot/graphs/contributors)