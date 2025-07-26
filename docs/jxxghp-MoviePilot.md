# MoviePilot: Automate Your Movie and TV Show Management

MoviePilot is a powerful and user-friendly application designed to streamline your movie and TV show library management. For more details, visit the original repository: [MoviePilot on GitHub](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

## Key Features

*   **Modern Architecture:** Built with a clean and efficient architecture based on FastAPI (backend) and Vue3 (frontend).
*   **Simplified Design:** Focuses on core automation needs, minimizing complexity for easier use and maintenance.
*   **Intuitive User Interface:** Features a redesigned, user-friendly interface for a better experience.
*   **Frontend/Backend Separation:** Separate frontend and backend for easier development and maintenance.
*   **API Documentation:**  Accessible API documentation via `http://localhost:3001/docs`.
*   **Plugin Support:**  Extensible through a plugin architecture, allowing for customized functionality.

## Installation and Usage

For detailed installation and usage instructions, please refer to the official wiki: [MoviePilot Wiki](https://wiki.movie-pilot.org).

## Development

### Prerequisites

*   Python 3.12
*   Node JS v20.12.1

### Steps

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the platform-specific library files (`.so`, `.pyd`, or `.bin`) from `MoviePilot-Resources/resources` to `MoviePilot/app/helper`.

3.  **Install Backend Dependencies and Run:**

    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend will run on port `3001`.

4.  **Clone and Run the Frontend Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    The frontend will be accessible at `http://localhost:5173`.

5.  **Plugin Development:** Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) for creating custom plugins in the `app/plugins` directory.

## Contributing

Contributions are welcome!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>

***
**Disclaimer:** This project is for learning and discussion purposes only. Do not promote this project on any domestic platforms.