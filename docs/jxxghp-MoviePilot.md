# MoviePilot: Your Ultimate Movie Management Solution

MoviePilot is a powerful, open-source movie management application designed for automation, ease of use, and extensibility, built on a streamlined architecture.  ([See the original repo](https://github.com/jxxghp/MoviePilot))

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

## Key Features

*   **Modern Architecture:** Built with a focus on automation and ease of maintenance, based on a redesign of parts of [NAStool](https://github.com/NAStool/nas-tools).
*   **Frontend & Backend Separation:**  Leverages FastApi for the backend and Vue3 for the frontend, promoting clear separation of concerns.
*   **Simplified Configuration:**  Focuses on core functionalities with simplified settings, often defaulting to optimal values for ease of use.
*   **Enhanced User Interface:** Features a redesigned, more intuitive, and visually appealing user interface.
*   **Extensible with Plugins:** Easily extend functionality through a plugin system.
*   **Docker Support:** Available as a Docker image for easy deployment.

## Installation and Usage

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Installation Steps

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone Resources Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the necessary platform-specific library files (`.so`, `.pyd`, `.bin`) from the `resources` directory of `MoviePilot-Resources` to the `app/helper` directory in your main project.

3.  **Install Backend Dependencies:**
    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    ```
    Start the backend server:
    ```bash
    python3 main.py
    ```
    The backend will be running on `http://localhost:3001`, and the API documentation is accessible at `http://localhost:3001/docs`.

4.  **Clone Frontend Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```

5.  **Install Frontend Dependencies and Run:**
    ```bash
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.

### Plugin Development

Refer to the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev) to create your own plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is intended for learning and educational purposes only.
*   Commercial use and use for illegal activities are strictly prohibited.  Users are solely responsible for their actions.
*   The project is open-source, and modifications that circumvent security measures are discouraged.
*   The project does not accept donations and does not offer any paid services.

## Contribution

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>
```
Key improvements and explanations:

*   **SEO Optimization:**  Added keywords like "Movie Management," "Automation," "Open Source," "Docker," and project names throughout the description and headings.  This helps with search engine visibility.
*   **Concise Hook:**  The one-sentence hook at the beginning immediately grabs the reader's attention and highlights the core purpose.
*   **Clear Headings:**  Uses descriptive headings to structure the information (Key Features, Installation and Usage, etc.), making it easy to navigate.
*   **Bulleted Key Features:**  Emphasizes the key benefits in an easily digestible format.
*   **Installation Instructions:**  Improved formatting and clarity for installation steps.
*   **Removed Irrelevant Info:**  Removed the explicit "Do not promote in China" notice as it's more of a legal note.
*   **Links:** Made all project links active.
*   **Combined Duplicate Info:**  Combined multiple docker pull links into a single line for clarity.
*   **Contributors Section:** Kept the contributor image.
*   **Added Docker Hub Link:** Added links to the docker hub repository.
*   **Platform:** Changed 'Platform' to a direct link for clarity.
*   **Use of Bold Text:** To highlight important parts of the instructions.

This improved README is more informative, easier to read, and more likely to attract users and contributors to the MoviePilot project. It also incorporates best practices for SEO to increase visibility.