# MoviePilot: Your Automated Movie Management Solution

MoviePilot is a powerful and user-friendly application designed to automate your movie management workflow, offering a streamlined experience for users.  ([Original Repository](https://github.com/jxxghp/MoviePilot))

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

## Key Features

*   **Modern Architecture:** Built with a front-end (Vue3) and back-end (FastAPI) separation for enhanced performance and maintainability.
*   **Simplified User Experience:** Focuses on core automation needs, minimizing complexity and offering sensible defaults for easy setup.
*   **Intuitive User Interface:**  A redesigned and user-friendly interface makes managing your movie library a breeze.
*   **Extensible with Plugins:**  Plugin support to add custom functionality and expand the capabilities.

## Installation and Usage

Detailed installation and usage instructions are available in the official wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Quick Start (Local Development)

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the required platform-specific library files (`.so`, `.pyd`, `.bin`) from the `MoviePilot-Resources/resources` directory to the `app/helper` directory within your main project.

3.  **Install Backend Dependencies and Run the Backend:**
    ```bash
    cd <your_project_directory>/MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend server will start on port 3001.  API documentation is available at `http://localhost:3001/docs`.

4.  **Clone and Run the Frontend:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    The frontend will be accessible at `http://localhost:5173`.

5.  **Develop Plugins:** Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) within the wiki to create and integrate custom plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is for learning and educational purposes only. It is not intended for commercial use or illegal activities.
*   The developers are not responsible for the user's actions. Use this software at your own risk.
*   The source code is open-source.  Modifications and redistribution that violate the original intent are the sole responsibility of the redistributor.
*   The project does not accept donations.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>