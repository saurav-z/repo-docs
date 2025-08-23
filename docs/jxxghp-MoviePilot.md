# MoviePilot: Your Ultimate Automation Companion for Media Management

MoviePilot is a powerful and user-friendly application designed to streamline your media management, offering a clean, efficient, and easily extensible solution.  [Check out the original repository](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

**Key Features:**

*   **Modern Architecture:** Built with a front-end (Vue3) and back-end (FastAPI) separation for enhanced maintainability and scalability.
*   **Simplified Design:** Focuses on core automation needs, simplifying configurations with sensible defaults for ease of use.
*   **Intuitive User Interface:** Enjoy a redesigned, user-friendly interface for a more pleasant experience.
*   **Extensible with Plugins:**  Expand functionality by creating and integrating custom plugins.

## Getting Started

### Installation and Usage

1.  **Prerequisites:** Ensure you have `Python 3.12` and `Node JS v20.12.1` installed.
2.  **Clone the Main Project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

3.  **Clone Resource Project:** Clone the [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources) repository and copy the necessary platform-specific libraries (`.so`, `.pyd`, `.bin`) from the `resources` directory to the `app/helper` directory within the main project.
4.  **Install Backend Dependencies and Run:**  Navigate to the `app` directory and install the backend dependencies.  Then, run the `main.py` file to start the backend service. The default API port is `3001`.  Access the API documentation at `http://localhost:3001/docs`.

    ```bash
    cd MoviePilot/app
    pip install -r requirements.txt
    python3 main.py
    ```

5.  **Clone Frontend Project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```

6.  **Install Frontend Dependencies and Run:** Navigate to the frontend project directory and install the dependencies.  Then, start the frontend server and access it at `http://localhost:5173`.

    ```bash
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```

7.  **Plugin Development:** Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins within the `app/plugins` directory.

### Additional Resources

*   **Official Wiki:**  [https://wiki.movie-pilot.org](https://wiki.movie-pilot.org)
*   **API Documentation:** [https://api.movie-pilot.org](https://api.movie-pilot.org)

## Contributing

We welcome contributions!  See the project's contribution guidelines for more information.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is for learning and communication purposes only.
*   It should not be used for commercial or illegal activities.
*   The developers are not responsible for user actions.
*   The software is open-source, and modifications that bypass restrictions are not recommended.
*   This project does not accept donations and does not offer paid services.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>