# MoviePilot: Automate Your Movie and Media Workflow

MoviePilot is a powerful, streamlined, and user-friendly solution designed to automate your movie and media management tasks.  Check out the original project on [GitHub](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

## Key Features

*   **Frontend & Backend Separation:**  Built with a modern architecture using FastAPI for the backend and Vue3 for the frontend, ensuring a responsive and maintainable system.
*   **Simplified Configuration:**  Focuses on core automation needs, reducing complexity with sensible defaults and streamlined settings.
*   **User-Friendly Interface:**  Features a redesigned, intuitive, and aesthetically pleasing user interface for an improved experience.
*   **Extensible:** Supports custom plugins.

## Installation and Usage

Detailed installation and usage instructions can be found on the [official MoviePilot Wiki](https://wiki.movie-pilot.org).

## Development

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Steps

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the necessary `.so`/`.pyd`/`.bin` files from the `MoviePilot-Resources/resources` directory to `MoviePilot/app/helper`.

3.  **Install Backend Dependencies and Run:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend server will run on port 3001 by default.  API documentation is available at `http://localhost:3001/docs`.

4.  **Clone and Run the Frontend Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.

5.  **Plugin Development:**  Refer to the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is for learning and educational purposes only.
*   Do not use this software for commercial purposes or illegal activities.
*   The software's developers are not responsible for user actions.
*   The project is open-source; modifications that bypass security measures are discouraged.
*   The project does not accept donations.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>