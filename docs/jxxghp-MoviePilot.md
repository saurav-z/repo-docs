# MoviePilot: Your Automated Movie Management Solution

MoviePilot is a powerful and streamlined application designed to automate your movie management workflow.  (Check out the original repository [here](https://github.com/jxxghp/MoviePilot)!)

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/network/members)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/graphs/contributors)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/archive/refs/heads/main.zip)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

*Based on code from [NAStool](https://github.com/NAStool/nas-tools), MoviePilot focuses on automating core movie management tasks, offering a simplified and extensible solution.*

**Important Notice:** This project is intended for learning and discussion purposes only. Please refrain from promoting this project on any domestic platforms.

**Stay updated:** Join our Telegram channel: [https://t.me/moviepilot_channel](https://t.me/moviepilot_channel)

## Key Features

*   **Frontend & Backend Separation:** Built with FastAPI (backend) and Vue3 (frontend) for a modern and maintainable architecture.
*   **Simplified Configuration:** Designed with a focus on essential features, minimizing complex settings with sensible defaults.
*   **Enhanced User Interface:**  Features a redesigned user interface for improved usability and a more pleasant experience.
*   **Extensible:** Supports plugin development to extend the functionalities.

## Getting Started

### Installation

Detailed installation instructions are available on the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Setup

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the necessary `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory to the `MoviePilot/app/helper` directory, matching your target platform and version.

3.  **Backend Setup:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend will run on port `3001` by default, with API documentation available at `http://localhost:3001/docs`.

4.  **Frontend Setup:**
    ```bash
    cd MoviePilot
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    The frontend will be accessible at `http://localhost:5173`.

5.  **Plugin Development:**
    Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>