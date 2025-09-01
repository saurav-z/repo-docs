# MoviePilot: Your All-in-One Movie Automation Hub

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a powerful and user-friendly application designed to automate and streamline your movie-related tasks, built with a focus on core automation needs and ease of use.

**[View the original repository on GitHub](https://github.com/jxxghp/MoviePilot)**

## Key Features

*   **Modern Architecture:** Built with a decoupled frontend (Vue3) and backend (FastAPI) for enhanced performance and maintainability.
*   **Simplified Configuration:** Focuses on essential features, reducing complexity and offering sensible default settings.
*   **Intuitive User Interface:** Features a redesigned UI for a more pleasant and efficient user experience.
*   **Docker Support:** Ready to run with Docker, simplifying deployment across different platforms.
*   **Extensible with Plugins:** Easily extend the functionality through custom plugins.

## Getting Started

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Installation

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the required `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory (matching your platform and version) to the `app/helper` directory within the main project.
3.  **Install Backend Dependencies:**
    ```shell
    cd MoviePilot
    pip install -r requirements.txt
    ```
4.  **Run the Backend:**
    *   Set `app` as the source code root. Run `main.py` to start the backend server. The default API documentation is available at `http://localhost:3001/docs`.
    ```shell
    python3 main.py
    ```
    The backend server will listen on port 3001.
5.  **Clone and Run the Frontend:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    *   Access the frontend at `http://localhost:5173`.
6.  **Plugin Development:**
    *   Consult the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) for instructions on creating custom plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Important Notes & Disclaimer

*   This software is intended for educational and personal use only.
*   Please refrain from using this project for any commercial activities or in any way that violates local laws or regulations.
*   The developers are not responsible for any misuse of this software.  Users are solely responsible for their actions and how they utilize this project.
*   The source code is open-source.  Any modifications or distributions should adhere to the guidelines and disclaimers outlined in the project.
*   The project does not accept donations or offer any paid services. Please exercise caution to avoid any potential scams or misrepresentations.

## Contributing

[See the contributor graph](https://github.com/jxxghp/MoviePilot/graphs/contributors)

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>