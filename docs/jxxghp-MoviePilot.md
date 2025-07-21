# MoviePilot: Your Ultimate Movie Automation Companion

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a powerful, streamlined solution for automating your movie collection management.  Built upon the foundation of [NAStool](https://github.com/NAStool/nas-tools), MoviePilot focuses on core automation, simplifying setup and enhancing maintainability.

**Disclaimer:** *This project is for learning and discussion purposes only. Please refrain from promoting this project on any platforms within China.*

**Stay Updated:** Join the MoviePilot community on Telegram: [https://t.me/moviepilot_channel](https://t.me/moviepilot_channel)

## Key Features

*   **Frontend & Backend Separation:** Built with FastAPI (Python) for the backend and Vue3 for the frontend, providing a clean and maintainable architecture.
*   **Simplified Configuration:**  Focuses on essential features with sensible defaults, making setup quick and easy.
*   **Enhanced User Interface:** A redesigned user interface for a more intuitive and enjoyable user experience.
*   **Docker Support:** Easily deploy and run MoviePilot using Docker containers.
*   **Plugin Architecture**: Extensible via plugin development.

## Installation and Usage

Detailed installation and usage instructions are available on the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Setup

1.  **Clone the Main Repository:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone Resources Repository:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the necessary `.so`, `.pyd`, or `.bin` files for your platform from the `MoviePilot-Resources/resources` directory into the `app/helper` directory of the main project.

3.  **Backend Setup:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend service will start on port `3001`.
    *   API Documentation: `http://localhost:3001/docs`

4.  **Frontend Setup:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    *   Access the frontend at: `http://localhost:5173`

5.  **Plugin Development:**
    *   Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Contributing

We welcome contributions!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>

**[View the original repository on GitHub](https://github.com/jxxghp/MoviePilot)**