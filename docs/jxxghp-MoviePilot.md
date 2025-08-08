# MoviePilot: Your Automated Movie Management Solution

**MoviePilot streamlines your movie management by automating key tasks and providing a user-friendly interface.**

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is designed with a focus on core automation needs, making it easier to extend and maintain. This project is built upon code from [NAStool](https://github.com/NAStool/nas-tools).

**Important Note:** This project is intended for learning and discussion purposes only. Please refrain from promoting this project on any domestic platforms.

Stay updated via the official channel: https://t.me/moviepilot_channel

## Key Features

*   **Frontend and Backend Separation:** Built with FastApi (backend) and Vue3 (frontend). See the frontend project at [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend).
*   **Focus on Core Functionality:** Streamlined features and settings to simplify configuration, with many default options available.
*   **Modern User Interface:**  A redesigned, more user-friendly and visually appealing interface.
*   **Docker Support:** Easily deployable using Docker containers.

## Getting Started

### Installation & Usage

*   **Official Wiki:**  [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)
*   **API Documentation:** [https://api.movie-pilot.org](https://api.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node JS v20.12.1

### Installation Steps

1.  **Clone the Main Project:**

    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project:**

    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the required `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory (corresponding to your platform and version) into the `MoviePilot/app/helper` directory.

3.  **Backend Setup:**
    *   Navigate to the root directory of the cloned project.
    *   Install dependencies:

        ```shell
        pip install -r requirements.txt
        ```
    *   Run the backend server:

        ```shell
        python3 main.py
        ```

        The backend server defaults to port `3001`. Access the API documentation at `http://localhost:3001/docs`.

4.  **Frontend Setup:**
    *   Clone the frontend project:

        ```shell
        git clone https://github.com/jxxghp/MoviePilot-Frontend
        ```
    *   Install frontend dependencies and run the development server:

        ```shell
        yarn
        yarn dev
        ```
        Access the frontend at `http://localhost:5173`.

5.  **Plugin Development:** Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to develop plugins in the `app/plugins` directory.

## Contribution

We welcome contributions!  Check out the contributor list:

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>

## Resources

*   **[MoviePilot GitHub Repository](https://github.com/jxxghp/MoviePilot)** (Back to Original Repo)