# MoviePilot: Automate Your Movie and TV Show Management

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/network/members)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/graphs/contributors)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined, user-friendly application designed to automate your movie and TV show management, built for ease of use and extensibility.  [View the original repository](https://github.com/jxxghp/MoviePilot).

**Disclaimer:** This project is for learning and discussion purposes only. Please refrain from promoting this project on any platforms within China.

## Key Features

*   **Frontend and Backend Separation:** Built with FastAPI (backend) and Vue3 (frontend), providing a responsive and modern user experience. Access the API documentation at `http://localhost:3001/docs`. Frontend project: [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend).
*   **Simplified and Focused:**  Concentrates on core automation needs, simplifying configurations and reducing potential issues. Many settings have sensible defaults for easy setup.
*   **Intuitive User Interface:**  Features a redesigned and visually appealing user interface for enhanced usability.

## Getting Started

For detailed installation and usage instructions, please refer to the official Wiki: [MoviePilot Wiki](https://wiki.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Installation Steps

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the necessary platform-specific library files (`.so`, `.pyd`, or `.bin`) from the `MoviePilot-Resources/resources` directory to the `app/helper` directory within the main project.

3.  **Install Backend Dependencies and Run:**

    *   Navigate to the project directory and install the required Python packages.
        ```shell
        pip install -r requirements.txt
        ```
    *   Run the `main.py` file to start the backend service. The server defaults to port `3001`. API documentation is available at `http://localhost:3001/docs`.
        ```shell
        python3 main.py
        ```

4.  **Clone and Run the Frontend Project:**

    *   Clone the frontend repository:
        ```shell
        git clone https://github.com/jxxghp/MoviePilot-Frontend
        ```
    *   Install frontend dependencies and start the development server.
        ```shell
        yarn
        yarn dev
        ```
    *   Access the frontend at `http://localhost:5173`.

5.  **Plugin Development**
    * Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>