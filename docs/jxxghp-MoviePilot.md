# MoviePilot: Automate Your Movie Workflow with Ease

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a streamlined and user-friendly solution designed for automating your movie management tasks. Inspired by [NAStool](https://github.com/NAStool/nas-tools), MoviePilot focuses on core automation needs, simplifying functionality for enhanced maintainability and extensibility.

**Important Disclaimer:**  This project is intended for learning and educational purposes only. **Please do not promote this project on any platform within China.**

*   **Stay Updated:** [Telegram Channel](https://t.me/moviepilot_channel)

## Key Features

*   **Modern Architecture:** Built with a clean separation of front-end (Vue3) and back-end (FastAPI) for improved maintainability and scalability.
*   **Focused Functionality:** Streamlined features and settings, with sensible defaults for ease of use.
*   **Intuitive User Interface:** Redesigned user interface for a more pleasant and efficient experience.
*   **Extensible with Plugins:** Easily extend functionality through a plugin system (see [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev)).

## Getting Started

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Installation and Setup

1.  **Clone the Main Project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the Resources Project (for required libraries):**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```

    *   Copy the platform-specific library files (`.so`, `.pyd`, `.bin`) from the `MoviePilot-Resources/resources` directory to the `app/helper` directory in the main project.

3.  **Install Backend Dependencies:**

    ```bash
    cd MoviePilot/app
    pip install -r requirements.txt
    ```

4.  **Run the Backend:**

    ```bash
    python3 main.py
    ```

    *   The backend will run on port 3001 by default.
    *   Access the API documentation at `http://localhost:3001/docs`.

5.  **Clone the Frontend Project:**

    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    ```

6.  **Install Frontend Dependencies and Run the Frontend:**

    ```bash
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```

    *   Access the frontend at `http://localhost:5173`.

7.  **Plugin Development:**
    *   Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Legal and Ethical Considerations

**Disclaimer:** This software is for educational and personal use only. The developers are not responsible for any misuse or illegal activities performed with this software.

*   **Use at Your Own Risk:** The use of this software is entirely at your own risk.
*   **No Commercial Use:** This software should not be used for commercial purposes.
*   **No Responsibility:** The developers are not responsible for any actions taken by users.
*   **Open Source and Modification:**  The code is open source; however, any modifications that remove restrictions are the responsibility of the modifier.
*   **No Donations:** The project does not accept donations.

## Contribution

We welcome contributions from the community!

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>