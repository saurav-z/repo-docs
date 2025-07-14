# MoviePilot: Automate Your Movie and TV Show Management

[<img src="https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge" alt="GitHub stars"/>](https://github.com/jxxghp/MoviePilot)
[<img src="https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge" alt="GitHub forks"/>](https://github.com/jxxghp/MoviePilot)
[<img src="https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge" alt="GitHub contributors"/>](https://github.com/jxxghp/MoviePilot)
[<img src="https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge" alt="GitHub repo size"/>](https://github.com/jxxghp/MoviePilot)
[<img src="https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge" alt="GitHub issues"/>](https://github.com/jxxghp/MoviePilot)
[<img src="https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge" alt="Docker pulls"/>](https://hub.docker.com/r/jxxghp/moviepilot)
[<img src="https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge" alt="Docker pulls v2"/>](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[<img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge" alt="Platform"/>](https://github.com/jxxghp/MoviePilot)

MoviePilot is a powerful, streamlined application built to automate your movie and TV show management tasks, offering a user-friendly experience with a focus on core functionality.  Based on parts of [NAStool](https://github.com/NAStool/nas-tools), MoviePilot simplifies your media management workflow while being easily extensible.

**Important Note:** This project is for learning and communication purposes only. Please do not promote this project on any domestic platforms.

*   **Release Channel:** [Telegram Channel](https://t.me/moviepilot_channel)

## Key Features

*   **Frontend/Backend Separation:** Built with FastAPI (backend) and Vue3 (frontend) for a responsive and modern interface. Frontend available at [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend), API documentation:  `http://localhost:3001/docs`.
*   **Core Functionality Focus:** Simplifies features and settings, using sensible defaults for ease of use.
*   **Enhanced User Interface:** Redesigned user interface for a more intuitive and visually appealing experience.

## Installation and Usage

Detailed instructions and guides can be found on the official [MoviePilot Wiki](https://wiki.movie-pilot.org).

## Development

### Prerequisites
*   Python 3.12
*   Node JS v20.12.1

### Getting Started

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone Resources:** Clone the resources repository to obtain necessary platform-specific libraries.
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory (matching your platform and version) to the `MoviePilot/app/helper` directory.

3.  **Backend Setup:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend service will start on port 3001 by default, accessible via `http://localhost:3001/docs`.

4.  **Frontend Setup:**
    ```bash
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    The frontend will be accessible at `http://localhost:5173`.

5.  **Plugin Development:** Create plugins within the `app/plugins` directory by referring to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev).

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" alt="Contributors"/>
</a>