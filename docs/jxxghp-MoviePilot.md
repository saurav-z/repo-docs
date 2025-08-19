# MoviePilot: Automate Your Media Management with Ease

MoviePilot is a powerful media management solution designed for automation, offering a streamlined experience for users who want to simplify their media workflows.  Learn more and contribute on the original repository: [https://github.com/jxxghp/MoviePilot](https://github.com/jxxghp/MoviePilot)

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

## Key Features

*   **Modern Architecture:** Built with a front-end (Vue3) and back-end (FastAPI) separation for enhanced flexibility and maintainability.
*   **Simplified Configuration:** Focused on core functionalities with default settings, making it easy to set up and use.
*   **User-Friendly Interface:**  Features a redesigned user interface for a more intuitive and visually appealing experience.
*   **Extensible:** Supports plugin development for customized functionality.

## Getting Started

### Installation

Detailed instructions and setup guides are available on the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Quick Installation Guide

1.  **Clone the Main Repository:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resources Repository:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the platform-specific libraries (.so/.pyd/.bin) from the `MoviePilot-Resources/resources` directory to the `MoviePilot/app/helper` directory.

3.  **Install Backend Dependencies:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend service will start on port `3001`.
    *   API documentation is available at: `http://localhost:3001/docs`
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
    *   Access the frontend at: `http://localhost:5173`

### Plugin Development

Refer to the plugin development guide for creating custom plugins: [https://wiki.movie-pilot.org/zh/plugindev](https://wiki.movie-pilot.org/zh/plugindev)
Plugins should be placed in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Legal Disclaimer

*   This software is intended for educational and personal use only.
*   Commercial use or the use of this software for any illegal activities is strictly prohibited. Users are solely responsible for their actions.
*   The software is open-source and modifications that bypass security measures or violate user authentication are discouraged.
*   This project does not accept donations or provide paid services. Please be aware of potential scams.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>