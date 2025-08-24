# MoviePilot: Automated Media Management for Your Needs

**MoviePilot is a powerful and streamlined media management solution designed for automation, ease of use, and extensibility.** ([View the original repository on GitHub](https://github.com/jxxghp/MoviePilot))

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

Based on a redesign of code from [NAStool](https://github.com/NAStool/nas-tools), MoviePilot focuses on core automation needs, reducing complexity and improving maintainability and extensibility.

**Disclaimer: This project is intended for learning and discussion purposes only.  Please refrain from promoting this project on any platforms within China.**

Communication Channel: [MoviePilot Telegram Channel](https://t.me/moviepilot_channel)

## Key Features

*   **Frontend and Backend Separation:** Built with FastAPI (backend) and Vue3 (frontend) for a modern and responsive user experience.
*   **Core-Focused Design:** Simplifies functionality and settings, often with sensible defaults for ease of use.
*   **Improved User Interface:** A redesigned UI provides a more intuitive and aesthetically pleasing experience.

## Getting Started

For detailed installation and usage instructions, please refer to the official Wiki: [MoviePilot Wiki](https://wiki.movie-pilot.org)

## Development

### API Documentation

Access the API documentation at: [MoviePilot API Docs](https://api.movie-pilot.org)

### Prerequisites for Local Development

*   Python 3.12
*   Node.js v20.12.1

### Installation and Running

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the platform-specific library files (`.so`, `.pyd`, `.bin`) from the `resources` directory of `MoviePilot-Resources` to the `app/helper` directory of `MoviePilot`.

3.  **Install Backend Dependencies and Run the Backend:**
    ```shell
    cd MoviePilot/app
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend service will run on port `3001` by default.
    *   API Documentation will be available at `http://localhost:3001/docs`.

4.  **Clone and Run the Frontend Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    *   The frontend will be accessible at `http://localhost:5173`.

5.  **Plugin Development:**
    *   Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is for learning and communication purposes only. It is not intended for commercial use or illegal activities. Users are solely responsible for their actions.
*   This is open-source software. Modifying the code to remove restrictions and distributing it is the responsibility of the modifier. We do not recommend circumventing or changing the user authentication mechanism and distributing it publicly.
*   We do not accept donations and do not provide paid services. Please be aware of potential scams.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>