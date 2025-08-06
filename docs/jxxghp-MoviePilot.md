# MoviePilot: Your Automated Movie and Media Management Solution

MoviePilot is a powerful tool designed to automate your movie and media management, offering a streamlined and user-friendly experience. [Explore the original repository](https://github.com/jxxghp/MoviePilot) for more details.

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/network/members)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/graphs/contributors)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

**Important Note:** *This project is for learning and educational purposes only. Please refrain from promoting this project on any domestic platforms.*

## Key Features

*   **Frontend and Backend Separation:** Utilizes FastAPI for the backend and Vue3 for the frontend, providing a clean and efficient architecture.  Frontend project: [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend).
*   **Focus on Core Automation:** Designed to streamline key functionalities, minimizing complexity and offering easily manageable settings with sensible defaults.
*   **Enhanced User Interface:** A redesigned user interface for a more intuitive and visually appealing experience.

## Getting Started

*   **Official Wiki:** https://wiki.movie-pilot.org
*   **API Documentation:** https://api.movie-pilot.org

## Installation and Usage

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Project and copy required libraries:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources` project, corresponding to your platform and version, into the `app/helper` directory of the main project.
3.  **Install Backend Dependencies and Run:**
    *   Navigate into the `app` directory of the main project.
    ```shell
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend will run on port `3001` by default.  Access the API documentation at `http://localhost:3001/docs`.
4.  **Clone and Run the Frontend:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```
    Install frontend dependencies and run the frontend project. Access the frontend at `http://localhost:5173`.
    ```shell
    yarn
    yarn dev
    ```
5.  **Develop Plugins:**  Refer to the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev) to create custom plugins within the `app/plugins` directory.

## Contributing

We welcome contributions!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>