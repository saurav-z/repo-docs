# MoviePilot: Your Automated Movie and Media Management Solution

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined and user-friendly media management tool built for automation, offering a simplified approach to managing your movie and media library. Built upon a refactored architecture focused on core automation needs, MoviePilot provides a more maintainable and extensible solution.

## Key Features:

*   **Modern Architecture:** Built with FastAPI (backend) and Vue3 (frontend) for a responsive and efficient user experience.
*   **Simplified Design:** Focuses on essential functionality, reducing complexity and making setup easier.  Many default settings are provided for ease of use.
*   **Enhanced User Interface:** A redesigned user interface for a more intuitive and visually appealing experience.
*   **Backend API:** Accessible through http://localhost:3001/docs

## Installation and Usage

For detailed installation instructions and usage guides, please visit the official wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development Setup

**Prerequisites:**

*   Python 3.12
*   Node JS v20.12.1

**Steps:**

1.  **Clone the Main Repository:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone Resources Repository:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the appropriate platform-specific library files (.so/.pyd/.bin) from the `MoviePilot-Resources/resources` directory to the `MoviePilot/app/helper` directory.
3.  **Install Backend Dependencies:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend service will start on port 3001, and the API documentation is available at http://localhost:3001/docs.
4.  **Clone the Frontend Repository:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```
5.  **Install Frontend Dependencies and Run:**
    ```bash
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at http://localhost:5173.
6.  **Plugin Development:**
    Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) for information on creating plugins in the `app/plugins` directory.

## Contributing

We welcome contributions!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>

## Important Notice

**This project is intended for learning and educational purposes only. Please refrain from promoting or distributing this project on any platforms within China.**

## Get Involved

*   **Original Repository:** [https://github.com/jxxghp/MoviePilot](https://github.com/jxxghp/MoviePilot)
*   **Frontend Repository:** [https://github.com/jxxghp/MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   **Telegram Channel:** [https://t.me/moviepilot_channel](https://t.me/moviepilot_channel)