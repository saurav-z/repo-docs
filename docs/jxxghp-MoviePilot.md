# MoviePilot: Automate Your Movie and TV Show Management

MoviePilot is a powerful, streamlined application designed to automate your movie and TV show management, built with a focus on core needs and ease of use.  ([View the original repository](https://github.com/jxxghp/MoviePilot))

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/network/members)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/graphs/contributors)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

*Note: This project is for learning and discussion purposes only. Please do not promote this project on any domestic platforms.*

Stay updated: [Telegram Channel](https://t.me/moviepilot_channel)

## Key Features

*   **Modern Architecture:** Built with a frontend and backend separation using FastAPI (Python) and Vue3, providing a responsive and efficient user experience.
*   **Focused Design:** Prioritizes core automation requirements, simplifying features and settings for ease of use. Default settings are often sufficient.
*   **Enhanced User Interface:** Features a redesigned, more visually appealing, and intuitive user interface.
*   **Extensible:** Designed for easy expansion and maintenance.

## Getting Started

*   **Official Wiki:** [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)
*   **API Documentation:** [https://api.movie-pilot.org](https://api.movie-pilot.org)

## Installation and Setup

1.  **Clone the Main Project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone Resources (and copy appropriate files):**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```

    Copy the `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources`' `resources` directory to the `app/helper` directory of your main project, matching your platform and version.

3.  **Install Backend Dependencies:**

    ```bash
    cd <your_moviepilot_directory>/app  # Navigate to the app directory
    pip install -r requirements.txt
    ```

4.  **Run the Backend:**
    ```bash
    python3 main.py
    ```
    The backend service will start on port `3001`. Access API documentation at `http://localhost:3001/docs`.

5.  **Clone the Frontend Project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```

6.  **Run the Frontend:**

    ```bash
    cd <your_moviepilot_frontend_directory>
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.

7.  **Plugin Development:**  Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Contributing

We welcome contributions!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>