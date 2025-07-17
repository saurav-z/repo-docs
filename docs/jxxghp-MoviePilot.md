# MoviePilot: Your Automated Movie & Media Management Solution

MoviePilot simplifies your media management workflow, offering an intuitive interface and powerful automation capabilities.  Explore the project on [GitHub](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

## Key Features

*   **Modern Architecture:** Built with a frontend (Vue3) and backend (FastAPI) for a responsive and user-friendly experience.
*   **Simplified Configuration:** Focuses on core automation needs, reducing complexity and offering sensible defaults.
*   **Enhanced User Interface:** Features a redesigned, intuitive, and aesthetically pleasing user interface.
*   **API Documentation:**  Access the API documentation at `http://localhost:3001/docs`.

## Installation and Usage

For detailed instructions, consult the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development Setup

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
    Copy the necessary `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory to the `MoviePilot/app/helper` directory, based on your platform.

3.  **Install Backend Dependencies and Run:**

    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend will run on port 3001 by default.

4.  **Clone and Run the Frontend Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.

5.  **Plugin Development:** Develop custom plugins in the `app/plugins` directory, following the guidance provided in the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev).

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>