# MoviePilot: Your Automated Media Management Solution

MoviePilot is a powerful media management solution designed for automation, built upon a streamlined architecture for easy use and scalability. Explore the project on [GitHub](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

**Disclaimer:** This project is for educational purposes and personal use only.  Please refrain from promoting this project on any platforms in China.

Stay updated through our release channel: [Telegram Channel](https://t.me/moviepilot_channel)

## Key Features

*   **Modern Architecture:** Built with FastAPI (backend) and Vue3 (frontend) for a responsive and efficient user experience.
*   **Simplified Configuration:** Designed with a focus on core automation needs, minimizing complex settings for ease of use. Default values are available for many settings.
*   **Intuitive User Interface:** A redesigned user interface provides a more appealing and user-friendly experience.
*   **Frontend Repository:** [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   **API Documentation:** http://localhost:3001/docs

## Installation and Usage

For detailed installation and usage instructions, please refer to the official Wiki: [MoviePilot Wiki](https://wiki.movie-pilot.org)

## Development Setup

To contribute to MoviePilot, you will need:

*   Python 3.12
*   Node.js v20.12.1

Follow these steps to set up your development environment:

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone Resources Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the necessary `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory (matching your platform and version) into the `app/helper` directory of the main project.

3.  **Install Backend Dependencies and Run:**
    ```shell
    cd <path_to_moviepilot_project>
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend server will run on port 3001 by default.  API Documentation is accessible at: http://localhost:3001/docs

4.  **Clone and Run Frontend:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd <path_to_moviepilot_frontend_project>
    yarn
    yarn dev
    ```
    *   Access the frontend at http://localhost:5173

5.  **Plugin Development:**
    *   Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to develop plugins within the `app/plugins` directory.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>