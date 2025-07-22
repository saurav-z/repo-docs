# MoviePilot: Automate Your Movie and TV Show Management 

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/network/members)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/graphs/contributors)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined, efficient solution for managing your movie and TV show library, built for ease of use and extensibility.  [Check out the original repository here](https://github.com/jxxghp/MoviePilot).

**Important Note:** This project is for learning and educational purposes only. Please refrain from promoting this project on any domestic platforms.

## Key Features

*   **Frontend/Backend Separation:**  Uses FastApi (backend) and Vue3 (frontend) for a modern, maintainable architecture.  Frontend project available at [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend). API Documentation: http://localhost:3001/docs
*   **Core Functionality Focus:** Designed around essential automation needs, simplifying configurations and reducing potential issues.
*   **Simplified Configuration:** Defaults provided for many settings to ease setup.
*   **Enhanced User Interface:**  Redesigned UI for improved aesthetics and ease of navigation.
*   **Platform Compatibility:** Supports Windows, Linux, and Synology environments.
*   **Docker Support:** Available via Docker Hub for easy deployment and management.

## Installation and Usage

For detailed installation instructions, please refer to the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development

### Prerequisites:

*   Python 3.12
*   Node JS v20.12.1

### Steps:

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the necessary `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory (corresponding to your platform and version) to the `MoviePilot/app/helper` directory.

3.  **Install Backend Dependencies and Run:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend service will run on port `3001` by default. API documentation is available at `http://localhost:3001/docs`.

4.  **Clone and Run the Frontend:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    *   Access the frontend at `http://localhost:5173`.

5.  **Plugin Development:**
    *   Develop custom plugins within the `MoviePilot/app/plugins` directory. Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) for details.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>