# MoviePilot: Your Automated Movie Management Solution

MoviePilot streamlines your movie collection management with a focus on automation, ease of use, and extensibility.  [Visit the original repository on GitHub](https://github.com/jxxghp/MoviePilot) to explore its capabilities!

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/network/members)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/graphs/contributors)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

*Disclaimer: This project is for learning and communication purposes only. Please refrain from promoting it on any domestic platforms.*

**Stay updated:** [MoviePilot Channel on Telegram](https://t.me/moviepilot_channel)

## Key Features

*   **Modern Architecture:** Built with a frontend (Vue3) and backend (FastAPI) for a responsive and user-friendly experience. Frontend: [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend), API Docs: http://localhost:3001/docs
*   **Automated Core Functionality:**  Focuses on essential automation tasks, simplifying setup and configuration.
*   **Intuitive User Interface:**  Designed with a fresh, modern UI for improved usability.
*   **Extensible with Plugins:** Develop custom plugins to tailor MoviePilot to your specific needs.

## Getting Started

For detailed installation and usage instructions, please refer to the official Wiki: [MoviePilot Wiki](https://wiki.movie-pilot.org)

## Contributing

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Development Setup

1.  **Clone the Main Repository:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Repository:**  Acquire necessary platform-specific libraries.
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the appropriate `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory to `MoviePilot/app/helper`.

3.  **Install Backend Dependencies and Run:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend will run on port 3001 by default.  API documentation is available at http://localhost:3001/docs.

4.  **Clone and Run Frontend:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    The frontend will be accessible at http://localhost:5173.

5.  **Plugin Development:**  Create custom plugins within the `app/plugins` directory following the guidelines in the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev).

## Contributors

[<img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />](https://github.com/jxxghp/MoviePilot/graphs/contributors)