# MoviePilot: Your Ultimate Movie Management Solution

MoviePilot is a streamlined, open-source project designed to automate and simplify your movie management workflow.  ([Original Repo](https://github.com/jxxghp/MoviePilot))

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

## Key Features

*   **Modern Architecture:** Built with a frontend (Vue3) and backend (FastAPI) for a responsive and user-friendly experience.
*   **Simplified Configuration:**  Focuses on essential features, offering sensible defaults to minimize complex setup.
*   **Intuitive User Interface:**  Features a redesigned user interface for enhanced usability and a more aesthetically pleasing experience.
*   **Open Source and Extensible:**  Leverages an open source codebase that is easily expandable through the plugin system.
*   **Docker Support:** Easily deploy and run the application with Docker images.

## Getting Started

*   **Wiki:** Comprehensive guides and documentation are available on the [MoviePilot Wiki](https://wiki.movie-pilot.org).
*   **API Documentation:** Explore the API endpoints via [API documentation](https://api.movie-pilot.org)
*   **System Requirements**
    *   Python 3.12
    *   Node JS v20.12.1

### Installation & Usage

1.  **Clone the Main Project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the necessary `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory to the `app/helper` directory of the main project, based on your platform and version.

3.  **Install Backend Dependencies:**

    ```bash
    cd <MoviePilot_directory>/app  # Assuming 'app' is the root directory.
    pip install -r requirements.txt
    python3 main.py
    ```
    *   The backend service will start on port `3001` by default.
    *   Access the API documentation at `http://localhost:3001/docs`

4.  **Clone the Frontend Project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```

5.  **Install Frontend Dependencies and Run:**

    ```bash
    cd <MoviePilot-Frontend_directory>
    yarn
    yarn dev
    ```
    *   Access the frontend at `http://localhost:5173`

6.  **Plugin Development**
    *   Follow the guidelines outlined in the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create your custom plugins.  Place your plugin code in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is for educational and personal use only.
*   Do not use this software for commercial purposes or illegal activities.
*   The developers are not responsible for user actions.
*   Contributions are welcome, but avoid circumventing user authentication or making public distributions.
*   The project does not accept donations or offer paid services.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>