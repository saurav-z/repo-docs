# MoviePilot: Your Automation Hub for Movies and Media Management

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined and user-friendly application designed to automate your movie and media management tasks, built upon core automation principles. This project is based on parts of [NAStool](https://github.com/NAStool/nas-tools) to simplify features and provide easy extensibility.

## Key Features

*   **Frontend & Backend Separation:** Utilizes a modern architecture with FastAPI (backend) and Vue3 (frontend) for a responsive and efficient user experience.
*   **Focus on Core Automation:** Simplifies features and settings, with sensible defaults for ease of use.
*   **User-Friendly Interface:** Offers a redesigned, more intuitive, and aesthetically pleasing user interface.
*   **Cross-Platform Support:** Compatible with Windows, Linux, and Synology systems.

## Installation and Usage

### Prerequisites
*   Python 3.12
*   Node.js v20.12.1

### Steps:

1.  **Clone the Main Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project and Copy Libraries:**  Clone `MoviePilot-Resources` and copy the appropriate platform and version-specific libraries (`.so`/`.pyd`/`.bin`) from the `resources` directory into the `app/helper` directory.
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```

3.  **Install Backend Dependencies and Run Backend:**
    *   Navigate to the `app` directory.
    ```shell
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend will start on port 3001 by default.  API documentation is available at `http://localhost:3001/docs`.

4.  **Clone and Run Frontend Project:**
    ```shell
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    Access the frontend at `http://localhost:5173`.

5.  **Plugin Development:** Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create custom plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Important Notices

*   **Disclaimer:** This software is intended for learning and personal use only. Do not use it for commercial purposes or any illegal activities. Users are solely responsible for their actions.
*   **Open Source:** The source code is open.  Modifying and redistributing the code (e.g., by removing restrictions) carries risks and responsibilities that rest solely with the modifier.  Do not circumvent or modify the user authentication mechanism.
*   **No Donations:** The project does not accept donations or offer paid services. Please be aware of potential scams.

## Contribution

Feel free to contribute!

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>

## Additional Notes for SEO Optimization:

*   **Keywords:**  Include relevant keywords throughout the README, such as "movie automation," "media management," "NAS," "FastAPI," "Vue3," and platform names (Windows, Linux, Synology).
*   **Headings:** Use clear and descriptive headings to organize information.
*   **Concise Language:** Keep sentences and paragraphs concise and easy to understand.
*   **Links:** Provide links to the original repository and related projects to improve discoverability.
*   **Alt Text:** Ensure images have descriptive alt text.
*   **Up-to-Date:** Regularly update the README to reflect project changes and new features.
*   **Docker:** Highlight docker support