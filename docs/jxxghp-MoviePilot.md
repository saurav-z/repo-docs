# MoviePilot: Automate Your Movie Experience (Learn More on GitHub!)

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined and user-friendly application, designed to automate and enhance your movie-watching experience.  Built on the foundation of [NAStool](https://github.com/NAStool/nas-tools), MoviePilot focuses on core automation needs, offering a simplified and extensible approach to movie management.

**[View the original project on GitHub](https://github.com/jxxghp/MoviePilot)**

## Key Features

*   **Modern Architecture:**  Built with a front-end and back-end separation using FastApi and Vue3 for enhanced performance and user experience.
*   **Simplified Design:** Focuses on essential features, streamlining configuration and offering sensible defaults for ease of use.
*   **Intuitive Interface:**  Features a redesigned, aesthetically pleasing, and user-friendly interface.
*   **Extensible with Plugins:** Supports plugin development for customization and advanced functionality.
*   **Docker Support:** Ready to deploy with Docker.

## Installation and Usage

For detailed instructions, refer to the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

### Prerequisites

*   Python 3.12
*   Node JS v20.12.1

### Getting Started (Local Development)

1.  **Clone the main project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the resources project and copy the necessary files:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```

    Copy the `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory (matching your platform and version) into the `app/helper` directory of your main project.

3.  **Install backend dependencies and run the backend:**  Navigate to the `app` directory and run `main.py`. The API will be accessible at `http://localhost:3001/docs`.

    ```bash
    cd app
    pip install -r requirements.txt
    python3 main.py
    ```

4.  **Clone and run the frontend:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```

    The frontend will be available at `http://localhost:5173`.

5.  **Plugin Development:**  Learn how to develop plugins in the `app/plugins` directory by referring to the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev).

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is intended for learning and personal use only.
*   Do not use this software for commercial purposes or illegal activities.
*   The developers are not responsible for user actions.
*   This project is open-source. Modifications that circumvent restrictions are discouraged.
*   This project does not accept donations and does not offer any paid services.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>