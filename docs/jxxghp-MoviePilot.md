# MoviePilot: Automate Your Movie & Media Management

[View the original MoviePilot repository on GitHub](https://github.com/jxxghp/MoviePilot).

![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)
![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)
![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)
![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)
![Docker Pulls V2](https://img.shields.io/docker/pulls/moviepilot-v2?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)

MoviePilot is a streamlined and user-friendly application designed for automating your movie and media management, built upon the foundation of [NAStool](https://github.com/NAStool/nas-tools) with a focus on core automation needs.

**Disclaimer:** *This project is for educational and collaborative purposes only. Please refrain from promoting this project on any platforms within China.*

## Key Features

*   **Frontend & Backend Separation:** Built with FastAPI (backend) and Vue3 (frontend) for a modern and responsive user experience.  Frontend Project: [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend). API Documentation: `http://localhost:3001/docs`
*   **Simplified Design:**  Focused on essential features, with simplified configurations and default settings for ease of use.
*   **Improved User Interface:** A redesigned UI provides a more aesthetically pleasing and intuitive experience.

## Installation & Usage

Detailed installation and usage instructions are available in the official Wiki: [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)

## Development

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Setup

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone Resources:** Clone the resources project and copy the required files.
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Then, copy the `.so`, `.pyd`, or `.bin` files from the `resources` directory (corresponding to your platform and version) to the `app/helper` directory within the main project.
3.  **Install Backend Dependencies:**
    ```shell
    cd <project_directory>/app  # Navigate to the 'app' directory.
    pip install -r requirements.txt
    python3 main.py # Starts the backend server (default port: 3001)
    ```
    The API documentation is accessible at: `http://localhost:3001/docs`
4.  **Clone the Frontend Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```
5.  **Install Frontend Dependencies and Run:**
    ```shell
    cd <project_directory>/MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at: `http://localhost:5173`
6.  **Plugin Development:** Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create custom plugins within the `app/plugins` directory.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>