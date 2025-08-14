# MoviePilot: Automate Your Media Workflow with a Powerful and User-Friendly Solution

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a powerful and user-friendly media management solution designed for automation and ease of use, built upon the foundation of [NAStool](https://github.com/NAStool/nas-tools). This project is focused on simplifying media workflows and provides a streamlined experience for users.

**[➡️ View the original repository on GitHub](https://github.com/jxxghp/MoviePilot)**

## Key Features

*   **Frontend & Backend Separation:** Utilizes FastAPI for the backend and Vue3 for the frontend, providing a clean and maintainable architecture.
*   **Simplified Configuration:** Focuses on core automation needs, minimizing complex settings and offering sensible defaults.
*   **Modern User Interface:** Features a redesigned, intuitive, and visually appealing user interface for a better user experience.
*   **Docker Support:** Offers Docker images for easy deployment and portability.
*   **Extensible with Plugins:** Allows users to extend functionalities through plugin development.

## Installation and Usage

For detailed installation and usage instructions, please refer to the official Wiki: [MoviePilot Wiki](https://wiki.movie-pilot.org).

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Getting Started

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the required `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory (matching your platform and version) into the `app/helper` directory of the main project.

3.  **Install Backend Dependencies:**
    ```bash
    cd <MoviePilot Project Root>
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend will start on port `3001` by default.  API Documentation:  `http://localhost:3001/docs`

4.  **Clone the Frontend Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```

5.  **Install Frontend Dependencies and Run:**
    ```bash
    cd <MoviePilot-Frontend Project Root>
    yarn
    yarn dev
    ```
    Access the frontend at: `http://localhost:5173`

6.  **Develop Plugins:**  Consult the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create custom plugins within the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is for learning and communication purposes only.  Do not use it for commercial gain or any illegal activities. Users are solely responsible for their actions.
*   The software is open-source. Modifying and redistributing the software with the intent to bypass any limitations or restrictions are not recommended, and the modifier bears full responsibility.
*   Contributions are welcome, but donations are not accepted.  Be wary of any requests for monetary contributions.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>