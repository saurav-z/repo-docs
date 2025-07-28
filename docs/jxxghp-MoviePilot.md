# MoviePilot: Automate Your Movie Collection with Ease

MoviePilot is a powerful, streamlined application designed to automate and manage your movie collection, built with a focus on simplicity and extensibility. Explore the project on [GitHub](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

**Important Note:** This project is intended for learning and discussion purposes only. Please refrain from promoting this project on any platforms within China.

## Key Features

*   **Modern Architecture:** Built with a modern tech stack using FastAPI (backend) and Vue.js 3 (frontend).
*   **Simplified Design:** Focuses on core automation needs, offering simplified settings and defaults for ease of use.
*   **Enhanced User Interface:** Features a redesigned, intuitive, and visually appealing user interface.
*   **Docker Support:** Available via Docker, simplifying deployment across various platforms.
*   **Extensible Plugin Architecture:** Supports plugin development to extend functionality.

## Getting Started

For detailed installation and usage instructions, please refer to the official [MoviePilot Wiki](https://wiki.movie-pilot.org).

## Development

To contribute to MoviePilot, you will need: `Python 3.12` and `Node JS v20.12.1`

### Setup

1.  **Clone the main project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the resources project:**  Download required `.so`, `.pyd`, or `.bin` files for your platform and place them in the `app/helper` directory.

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```

3.  **Install backend dependencies and run the server:**

    ```bash
    cd MoviePilot
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend server will be running on `http://localhost:3001`, and the API documentation can be accessed at `http://localhost:3001/docs`.

4.  **Clone the frontend project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```

5.  **Install frontend dependencies and run the frontend:**

    ```bash
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    The frontend will be accessible at `http://localhost:5173`.

6.  **Plugin Development:**  Follow the [plugin development guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Contributing

We welcome contributions!  See the [GitHub Repository](https://github.com/jxxghp/MoviePilot) for more details.

### Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>