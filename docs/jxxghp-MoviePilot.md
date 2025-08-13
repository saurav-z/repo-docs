# MoviePilot: Your Ultimate Automation Solution for Media Management

[Original Repository](https://github.com/jxxghp/MoviePilot)

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a reimagined media management tool built for automation, offering a streamlined experience for users seeking efficiency and ease of use.

## Key Features

*   **Frontend & Backend Separation:** Built on FastApi + Vue3 for a modern, responsive user interface.
*   **Focused Core Functionality:** Simplifies features and settings, making the platform easier to understand and use.
*   **Enhanced User Interface:** Redesigned UI for a more intuitive and visually appealing experience.
*   **Extensible with Plugins:** Supports plugin development for extending functionality.

## Installation & Usage

For detailed instructions, please refer to the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

### Prerequisites

*   Python 3.12
*   Node JS v20.12.1

### Installation Steps

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Project:**  Get necessary platform-specific libraries.
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the correct `.so`/`.pyd`/`.bin` files from the `MoviePilot-Resources/resources` directory to `MoviePilot/app/helper`.
3.  **Install Backend Dependencies:**
    ```bash
    cd MoviePilot/app
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend service runs on port `3001` by default.
    *   API Documentation: `http://localhost:3001/docs`
4.  **Clone the Frontend Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```
5.  **Install Frontend Dependencies & Run:**
    ```bash
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    *   Access the frontend at `http://localhost:5173`

6.  **Develop Plugins:**
    *   Consult the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev) for creating custom plugins in the `MoviePilot/app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is for learning and communication purposes only. Do not use it for commercial purposes or illegal activities.  Users are solely responsible for their actions.
*   This project is open source. Users are responsible for any modifications and resulting issues.
*   The project does not accept donations.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>