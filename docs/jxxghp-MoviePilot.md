# MoviePilot: Your Automated Movie & TV Show Companion

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined and user-friendly application designed to automate your movie and TV show management, built on a foundation of core automation needs.

**Please Note:** This project is for learning and communication purposes only. Do not promote this project on any domestic platforms.

*   **Join the Community:** [MoviePilot Channel (Telegram)](https://t.me/moviepilot_channel)

## Key Features

*   **Modern Architecture:** Separated frontend (Vue3) and backend (FastAPI) for enhanced performance and maintainability.  Frontend available at [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend), and API documentation at `http://localhost:3001/docs`.
*   **Focused Automation:**  Concentrates on essential automation tasks, simplifying configurations and offering sensible defaults.
*   **Intuitive User Interface:**  Features a redesigned, user-friendly interface for a superior user experience.
*   **Easy to Extend:**  Designed with extensibility in mind, making it simple to add new features and integrations via plugins.

## Installation and Usage

For detailed installation and usage instructions, please refer to the official wiki: [MoviePilot Wiki](https://wiki.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Setup

1.  **Clone the Main Project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project:** (Necessary for platform-specific libraries)

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the relevant `.so`, `.pyd`, or `.bin` files from `MoviePilot-Resources/resources/` to `MoviePilot/app/helper/`, based on your platform and version.

3.  **Backend Setup:**

    *   Navigate to the `MoviePilot/` directory.
    *   Install dependencies:

        ```bash
        pip install -r requirements.txt
        ```
    *   Run the backend server:

        ```bash
        python3 main.py
        ```
        The backend will run on port `3001` by default.  Access the API documentation at `http://localhost:3001/docs`.

4.  **Frontend Setup:**

    *   Clone the frontend repository:

        ```bash
        git clone https://github.com/jxxghp/MoviePilot-Frontend
        ```
    *   Install frontend dependencies and run the development server:

        ```bash
        cd MoviePilot-Frontend
        yarn
        yarn dev
        ```
        Access the frontend at `http://localhost:5173`.

5.  **Plugin Development:**

    *   Consult the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) in the wiki.
    *   Develop your plugins within the `app/plugins/` directory.

## Contributing

We welcome contributions!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>

**[Back to the Top](#moviepilot-your-automated-movie--tv-show-companion)**